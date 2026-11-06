import time
import random
import numpy as np

N = 16
NUM_POINTS = N * N
POINTS = [(x, y) for y in range(N) for x in range(N)]

# Precompute squared distances
coords = np.array(POINTS, dtype=np.int32)
diffs = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
DIST_SQ = np.sum(diffs * diffs, axis=-1)

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    best_size = NUM_POINTS + 1
    best_set = None
    
    def is_valid_addition(S_indices, p):
        """
        Check if p can be added to S_indices.
        S_indices is a list of indices.
        """
        if not S_indices:
            return True
        
        # Distances from p to all in S
        dists = DIST_SQ[p, S_indices]
        
        # 1. Distances must be unique
        if len(dists) != len(np.unique(dists)):
            return False
        
        # 2. For each b in S, d(p, b) must not have been "used" by b for another point in S.
        # We need to check if any pair (b, c) in S has d(b, c) == d(p, b).
        # This is equivalent to checking if d(p, b) is in the set of distances from b to S\{b}.
        # We can check this by: for each b in S, is there c in S (c!=b) such that DIST_SQ[b, c] == DIST_SQ[p, b]?
        
        # Vectorized check:
        # For each b in S_indices, we want to know if DIST_SQ[b, S_indices] contains DIST_SQ[p, b] at any index other than b itself.
        # Note: DIST_SQ[b, b] = 0. d(p, b) > 0.
        
        # Create a mask of distances from S to S
        # This is O(|S|^2). Since |S| is small (<20), this is fine.
        
        # Efficient check:
        # For each b in S_indices:
        #   d_pb = DIST_SQ[p, b]
        #   Check if any c in S_indices (c != b) has DIST_SQ[b, c] == d_pb
        #   This is equivalent to: (DIST_SQ[b, S_indices] == d_pb).any()
        #   But we must exclude c=b? DIST_SQ[b, b] = 0. d_pb > 0. So no conflict with c=b.
        
        for i, b in enumerate(S_indices):
            d_pb = dists[i]
            # Check if d_pb exists in distances from b to S
            # We can use np.isin or just check
            # Since |S| is small, a loop or simple numpy op is fine
            if np.any(DIST_SQ[b, S_indices] == d_pb):
                return False
        
        return True

    def construct_greedy(seed_val):
        random.seed(seed_val)
        order = list(range(NUM_POINTS))
        random.shuffle(order)
        
        S_indices = []
        
        for p in order:
            if is_valid_addition(S_indices, p):
                S_indices.append(p)
        
        # Maximality Repair
        # Check if any point not in S can be added
        changed = True
        while changed:
            changed = False
            for p in range(NUM_POINTS):
                if p in S_indices:
                    continue
                if is_valid_addition(S_indices, p):
                    S_indices.append(p)
                    changed = True
                    break # Restart scan after adding
            
        return S_indices

    def compute_blockers(S_indices):
        """
        For each x not in S, compute the set of pairs (a, b) in S that block x.
        A pair (a, b) blocks x if {a, b, x} is isosceles.
        Returns a dict: x -> set of (min(a,b), max(a,b))
        """
        S_set = set(S_indices)
        blockers = {}
        
        # Precompute all pairs in S
        pairs = []
        for i in range(len(S_indices)):
            for j in range(i + 1, len(S_indices)):
                pairs.append((S_indices[i], S_indices[j]))
        
        for x in range(NUM_POINTS):
            if x in S_set:
                continue
            
            blocking_pairs = set()
            for a, b in pairs:
                # Check if {a, b, x} is isosceles
                d_ab = DIST_SQ[a, b]
                d_ax = DIST_SQ[a, x]
                d_bx = DIST_SQ[b, x]
                
                if d_ax == d_bx or d_ax == d_ab or d_bx == d_ab:
                    blocking_pairs.add((a, b))
            
            blockers[x] = blocking_pairs
            
        return blockers

    def reduce_set(S_indices):
        """
        Try to remove points from S_indices while maintaining maximality.
        """
        S_indices = S_indices.copy()
        S_set = set(S_indices)
        
        # Initial blocker computation
        blockers = compute_blockers(S_indices)
        
        # To speed up removal checks, we track for each x, which pairs block it.
        # If we remove p, we remove all pairs containing p from blockers[x].
        # x becomes unblocked if blockers[x] becomes empty.
        
        improved = True
        while improved:
            improved = False
            # Shuffle S to try different removal orders
            S_copy = S_indices[:]
            random.shuffle(S_copy)
            
            for p in S_copy:
                if p not in S_set:
                    continue
                
                # Check if removing p keeps the set maximal
                # x (not in S) is unblocked by S\{p} iff all pairs blocking x contain p.
                # i.e., blockers[x] is a subset of { (a,b) in S | p in (a,b) }
                
                # Identify pairs containing p
                pairs_with_p = set()
                for a, b in [(S_indices[i], S_indices[j]) for i in range(len(S_indices)) for j in range(i+1, len(S_indices))]:
                    if a == p or b == p:
                        pairs_with_p.add((a, b))
                
                can_remove = True
                for x, b_pairs in blockers.items():
                    if not b_pairs:
                        # x is already unblocked? This shouldn't happen if S was maximal.
                        # But if it does, we can't remove p because x is already a problem?
                        # Actually, if b_pairs is empty, x is NOT blocked by S.
                        # This implies S is not maximal.
                        # But we assume S is maximal.
                        pass
                    
                    # x remains blocked by S\{p} iff there is a pair in b_pairs that does NOT contain p.
                    # If all pairs in b_pairs contain p, then removing p unblocks x.
                    # If x is unblocked, S\{p} is NOT maximal.
                    
                    # Check: Is there any pair in b_pairs that does NOT contain p?
                    is_still_blocked = False
                    for pair in b_pairs:
                        if p not in pair:
                            is_still_blocked = True
                            break
                    
                    if not is_still_blocked:
                        can_remove = False
                        break
                
                if can_remove:
                    # Remove p
                    S_indices.remove(p)
                    S_set.remove(p)
                    improved = True
                    
                    # Update blockers
                    # Remove all pairs containing p from blockers[x]
                    for x in blockers:
                        # Filter out pairs containing p
                        new_pairs = set()
                        for pair in blockers[x]:
                            if p not in pair:
                                new_pairs.add(pair)
                        blockers[x] = new_pairs
                    break # Restart loop with new S
            
        return S_indices

    # Main Loop
    attempts = 0
    while time.time() < deadline:
        attempts += 1
        
        # 1. Construct
        S = construct_greedy(attempts)
        
        # 2. Reduce
        S = reduce_set(S)
        
        current_size = len(S)
        if current_size < best_size:
            best_size = current_size
            best_set = S.copy()
            
            if best_size <= 10: # Very good, can stop early
                break

    if best_set is None:
        # Fallback
        best_set = list(range(NUM_POINTS))
        
    return [POINTS[i] for i in best_set]

if __name__ == "__main__":
    s = construct()
    print(len(s))
