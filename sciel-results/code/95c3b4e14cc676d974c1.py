"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 64
import time
import random
import numpy as np

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 285  # Safety margin
    
    # Grid points
    points = [(x, y) for y in range(N) for x in range(N)]
    num_points = N * N
    
    # Precompute squared distances
    xs = np.array([p[0] for p in points], dtype=np.int32)
    ys = np.array([p[1] for p in points], dtype=np.int32)
    
    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dist2 = (dx*dx + dy*dy).astype(np.int32)
    
    max_d2 = int(dist2.max())
    
    # used_dists[i][d] = 1 if distance d is used by point i
    used_dists = [bytearray(max_d2 + 1) for _ in range(num_points)]
    
    S = []
    S_set = set()
    
    def is_valid_add(P):
        """Check if P can be added to S."""
        if P in S_set:
            return False
        
        # Check 1: Distances from P to all Q in S must be unique
        seen_dists = set()
        for Q in S:
            d = int(dist2[P, Q])
            if d in seen_dists:
                return False
            seen_dists.add(d)
            
            # Check 2: Distance d must not be already used by Q
            if used_dists[Q][d]:
                return False
                
        return True

    def add_point(P):
        """Add P to S and update used_dists."""
        S.append(P)
        S_set.add(P)
        for Q in S:
            if Q != P:
                d = int(dist2[P, Q])
                used_dists[P][d] = 1
                used_dists[Q][d] = 1

    def remove_point(P):
        """Remove P from S and update used_dists."""
        S.remove(P)
        S_set.discard(P)
        for Q in S:
            d = int(dist2[P, Q])
            used_dists[P][d] = 0
            used_dists[Q][d] = 0
        # Clear P's used dists
        used_dists[P] = bytearray(max_d2 + 1)

    def greedy_construct(seed):
        """Run greedy construction with a given random seed. Returns the set S."""
        nonlocal S, S_set, used_dists
        # Reset state
        S.clear()
        S_set.clear()
        for i in range(num_points):
            used_dists[i] = bytearray(max_d2 + 1)
        
        rng = random.Random(seed)
        order = list(range(num_points))
        rng.shuffle(order)
        
        for P in order:
            if is_valid_add(P):
                add_point(P)
        
        return S[:]

    def local_search_with_removal(seed):
        """Local search that tries removal-and-refill strategy."""
        nonlocal S, S_set, used_dists
        rng = random.Random(seed)
        best_size = len(S)
        best_S = S[:]
        no_improve = 0
        max_no_improve = 200
        
        while time.time() < deadline - 15 and no_improve < max_no_improve:
            improved = False
            
            # Try to add points
            for _ in range(50):
                P = rng.randint(0, num_points - 1)
                if P not in S_set and is_valid_add(P):
                    add_point(P)
                    improved = True
                    break
            
            if not improved:
                # Try removal-and-refill: remove a point, then try to add multiple points
                if S:
                    # Pick a random point to remove
                    P_out = rng.choice(S)
                    remove_point(P_out)
                    
                    # Try to add as many points as possible
                    refill_count = 0
                    for _ in range(200):
                        P_in = rng.randint(0, num_points - 1)
                        if P_in not in S_set and is_valid_add(P_in):
                            add_point(P_in)
                            refill_count += 1
                    
                    if len(S) > best_size:
                        best_size = len(S)
                        best_S = S[:]
                        no_improve = 0
                    else:
                        no_improve += 1
                else:
                    no_improve += 1
            
            if improved:
                if len(S) > best_size:
                    best_size = len(S)
                    best_S = S[:]
                    no_improve = 0
        
        # Restore best state
        S.clear()
        S_set.clear()
        for i in range(num_points):
            used_dists[i] = bytearray(max_d2 + 1)
        for P in best_S:
            S.append(P)
            S_set.add(P)
        for i, P in enumerate(S):
            for j, Q in enumerate(S):
                if i != j:
                    d = int(dist2[P, Q])
                    used_dists[P][d] = 1
                    used_dists[Q][d] = 1
        
        return best_size

    # Phase 1: Multiple greedy restarts
    best_size = 0
    best_S = []
    
    # Try several different seeds for greedy construction
    for seed in range(5):
        if time.time() > deadline - 30:
            break
        S_candidate = greedy_construct(seed)
        if len(S_candidate) > best_size:
            best_size = len(S_candidate)
            best_S = S_candidate
    
    # Phase 2: Local search with removal-and-refill on best greedy result
    if best_S:
        # Set state to best_S
        S.clear()
        S_set.clear()
        for i in range(num_points):
            used_dists[i] = bytearray(max_d2 + 1)
        for P in best_S:
            S.append(P)
            S_set.add(P)
        for i, P in enumerate(S):
            for j, Q in enumerate(S):
                if i != j:
                    d = int(dist2[P, Q])
                    used_dists[P][d] = 1
                    used_dists[Q][d] = 1
        
        # Run local search
        local_search_with_removal(42)
        
        # Check if we improved
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
    
    # Phase 3: More local search iterations if time permits
    if time.time() < deadline - 20 and best_S:
        # Set state to best_S again
        S.clear()
        S_set.clear()
        for i in range(num_points):
            used_dists[i] = bytearray(max_d2 + 1)
        for P in best_S:
            S.append(P)
            S_set.add(P)
        for i, P in enumerate(S):
            for j, Q in enumerate(S):
                if i != j:
                    d = int(dist2[P, Q])
                    used_dists[P][d] = 1
                    used_dists[Q][d] = 1
        
        # Run more local search with different seed
        local_search_with_removal(123)
        
        if len(S) > best_size:
            best_size = len(S)
            best_S = S[:]
    
    return [points[i] for i in best_S]
# EVOLVE-BLOCK-END
