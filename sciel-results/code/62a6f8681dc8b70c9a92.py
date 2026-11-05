"""
Improved solution for No-isosceles set task.
Uses precomputed distance matrix and Simulated Annealing with Add/Remove/Swap moves.
"""
import time
import random
import math
import numpy as np

N = 64
TOTAL_POINTS = N * N

def construct():
    start_time = time.time()
    deadline = start_time + 285  # Safety margin for 300s limit

    # Generate all points
    points = [(x, y) for y in range(N) for x in range(N)]
    
    # Precompute squared distances
    # dist2[i, j] = squared distance between point i and point j
    # Using int32 to save memory, max dist2 is (63^2 + 63^2) * 2? No, (63-0)^2 + (63-0)^2 = 7938. Fits in int16, but int32 is safe and fast.
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    # Helper to index points
    def idx_to_point(idx):
        return points[idx]

    class NoIsoscelesState:
        def __init__(self):
            self.S = []  # List of indices in set
            self.S_set = set()  # For O(1) membership
            self.dist_sets = {}  # idx -> set of squared distances to other points in S
            
        def is_valid_add(self, q):
            if q in self.S_set:
                return False
            if not self.S:
                return True
                
            # Get distances from q to all p in S
            # Vectorized retrieval might be faster for large S, but S is small.
            # dist2[q, S] gives an array of distances.
            dists_q = dist2[q, self.S]
            
            # Check 1: Distances from q must be distinct
            # len(set(dists_q)) == len(self.S)
            if len(set(dists_q.tolist())) != len(self.S):
                return False
            
            # Check 2: For each p in S, dist2[p, q] must not be in dist_sets[p]
            # dist2 is symmetric, so dist2[p, q] == dist2[q, p]
            # dists_q[k] is dist2[q, S[k]]
            for k, p in enumerate(self.S):
                d = dists_q[k]
                if d in self.dist_sets[p]:
                    return False
            return True

        def add(self, q):
            if q in self.S_set:
                return False
            if not self.S:
                self.S.append(q)
                self.S_set.add(q)
                self.dist_sets[q] = set()
                return True
                
            dists_q = dist2[q, self.S].tolist()
            
            # We already checked validity, so we can assume it's valid
            # Update dist_sets for existing points
            for k, p in enumerate(self.S):
                self.dist_sets[p].add(dists_q[k])
                
            # Add q
            self.S.append(q)
            self.S_set.add(q)
            self.dist_sets[q] = set(dists_q)
            return True

        def remove(self, p):
            if p not in self.S_set:
                return False
            
            # Remove p from S
            # S is a list, remove is O(N). N is small.
            idx_in_list = self.S.index(p)
            self.S.pop(idx_in_list)
            self.S_set.discard(p)
            
            # Update dist_sets for remaining points
            # We need to remove the distance to p from each remaining point's set
            # dist2[remaining, p]
            # It is more efficient to iterate over current S (which is now S \ {p})
            # and remove dist2[remaining, p]
            # Note: dist2 is symmetric.
            
            # Store the list of remaining points before we modify anything else? 
            # self.S is already modified.
            for q in self.S:
                d = dist2[q, p]
                self.dist_sets[q].discard(d)
            
            del self.dist_sets[p]
            return True

        def size(self):
            return len(self.S)

    # Greedy Initialization
    def greedy_init(order):
        state = NoIsoscelesState()
        for idx in order:
            if state.is_valid_add(idx):
                state.add(idx)
        return state

    # Generate several random orders
    rng = random.Random(42)
    base_order = list(range(TOTAL_POINTS))
    
    best_state = None
    best_size = 0
    
    # 1. Random shuffles
    for _ in range(10):
        order = base_order[:]
        rng.shuffle(order)
        s = greedy_init(order)
        if s.size() > best_size:
            best_size = s.size()
            best_state = s

    # 2. Structured orders (e.g., diagonal, row-major)
    # Row-major
    s = greedy_init(list(range(TOTAL_POINTS)))
    if s.size() > best_size:
        best_size = s.size()
        best_state = s

    # Diagonal order (x+y)
    order_diag = sorted(range(TOTAL_POINTS), key=lambda i: points[i][0] + points[i][1])
    s = greedy_init(order_diag)
    if s.size() > best_size:
        best_size = s.size()
        best_state = s

    # Distance from center
    cx, cy = (N-1)/2.0, (N-1)/2.0
    order_center = sorted(range(TOTAL_POINTS), key=lambda i: (points[i][0]-cx)**2 + (points[i][1]-cy)**2)
    s = greedy_init(order_center)
    if s.size() > best_size:
        best_size = s.size()
        best_state = s

    if best_state is None:
        return []

    # Local Search (Simulated Annealing)
    current_state = NoIsoscelesState()
    # Copy best_state
    current_state.S = best_state.S[:]
    current_state.S_set = best_state.S_set.copy()
    current_state.dist_sets = {k: v.copy() for k, v in best_state.dist_sets.items()}
    
    best_state_SA = NoIsoscelesState()
    best_state_SA.S = current_state.S[:]
    best_state_SA.S_set = current_state.S_set.copy()
    best_state_SA.dist_sets = {k: v.copy() for k, v in current_state.dist_sets.items()}
    
    best_size = current_state.size()
    
    # Complement set for faster sampling? 
    # A list of indices not in S.
    # Updating a list complement is O(N) on remove. 
    # Given N=4096, and S is small, checking `if idx not in S_set` is O(1) but sampling uniformly from complement is hard without a list.
    # We can just sample random indices and check membership.
    # Probability of hitting complement is high if S is small.
    
    temp = 1.0
    cooling = 0.999
    min_temp = 0.01
    no_improve_count = 0
    max_no_improve = 2000
    
    # Precompute list of all indices for sampling
    all_indices = list(range(TOTAL_POINTS))
    
    while time.time() < deadline:
        # Check time frequently
        if time.time() > deadline:
            break
        
        action = rng.random()
        
        # 1. Add a random point (prob 0.4)
        if action < 0.4:
            # Sample a few random candidates to find a valid add
            added = False
            for _ in range(10):
                q = rng.choice(all_indices)
                if q not in current_state.S_set:
                    if current_state.is_valid_add(q):
                        current_state.add(q)
                        if current_state.size() > best_size:
                            best_size = current_state.size()
                            best_state_SA.S = current_state.S[:]
                            best_state_SA.S_set = current_state.S_set.copy()
                            best_state_SA.dist_sets = {k: v.copy() for k, v in current_state.dist_sets.items()}
                            no_improve_count = 0
                        else:
                            no_improve_count += 1
                        added = True
                        break
            if not added:
                no_improve_count += 1
                
        # 2. Remove a random point (prob 0.3)
        elif action < 0.7:
            if current_state.S:
                p = rng.choice(current_state.S)
                # Accept remove if it improves "flexibility"? 
                # Removing always decreases size, so we only do this to escape local minima.
                # We accept it with probability 1 (since it's a move in the search space)
                # but we track best separately.
                current_state.remove(p)
                no_improve_count += 1
                
        # 3. Swap a random point (prob 0.3)
        else:
            if current_state.S:
                # Try to swap p_out in S with p_in not in S
                swapped = False
                for _ in range(10):
                    p_out = rng.choice(current_state.S)
                    # Try to find a valid p_in
                    for _ in range(5):
                        p_in = rng.choice(all_indices)
                        if p_in in current_state.S_set or p_in == p_out:
                            continue
                        
                        # Check if p_in can be added if p_out were removed
                        # 1. Remove p_out
                        current_state.remove(p_out)
                        
                        # 2. Check if p_in is valid
                        if current_state.is_valid_add(p_in):
                            current_state.add(p_in)
                            if current_state.size() > best_size:
                                best_size = current_state.size()
                                best_state_SA.S = current_state.S[:]
                                best_state_SA.S_set = current_state.S_set.copy()
                                best_state_SA.dist_sets = {k: v.copy() for k, v in current_state.dist_sets.items()}
                                no_improve_count = 0
                            else:
                                no_improve_count += 1
                            swapped = True
                            break
                        else:
                            # Revert
                            current_state.add(p_out)
                            # Note: add(p_out) might fail if state changed? 
                            # We just removed it and didn't add anything else, so it should be valid to re-add.
                            # But is_valid_add is not checked in add. 
                            # We assume validity for re-add of removed point.
                            # To be safe, we can just restore state manually or trust the logic.
                            # Since we didn't modify dist_sets of others permanently except for the removal, 
                            # re-adding should restore them.
                            pass
                    
                    if swapped:
                        break
                if not swapped:
                    no_improve_count += 1

        # Cooling
        if no_improve_count > max_no_improve:
            # Restart from best
            current_state.S = best_state_SA.S[:]
            current_state.S_set = best_state_SA.S_set.copy()
            current_state.dist_sets = {k: v.copy() for k, v in best_state_SA.dist_sets.items()}
            no_improve_count = 0
            temp = 1.0
            
            # Optional: Perturb best state by removing a few points to explore nearby
            if current_state.S:
                num_remove = rng.randint(1, 3)
                to_remove = rng.sample(current_state.S, min(num_remove, len(current_state.S)))
                for p in to_remove:
                    current_state.remove(p)
        
        temp *= cooling
        if temp < min_temp:
            temp = min_temp

    # Return the best set found
    # best_state_SA is the state with max size
    return [idx_to_point(idx) for idx in best_state_SA.S]
