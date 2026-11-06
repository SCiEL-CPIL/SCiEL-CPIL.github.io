"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
from itertools import combinations

def construct():
    start_time = time.time()
    deadline = start_time + 85.0  # Safety margin before 90s limit
    
    N = 192
    MAX_K = 3
    
    def fast_greedy(order):
        """
        Greedy construction with optimized validity checks using auxiliary structures.
        K: Set of all sums of 3 distinct elements in S.
        R: Dict mapping element a to set of residues (u+v) % a for u,v in S\{a}, u!=v.
        """
        S = []
        K = set()  # Sums of 3 distinct elements
        R = {}     # Residues of pair sums mod a
        
        for x in order:
            valid = True
            
            # Check 1: x does not divide any sum of 3 distinct elements in S
            # If x divides sum(s), then sum(s) % x == 0
            for s in K:
                if s % x == 0:
                    valid = False
                    break
            
            # Check 2: For each a in S, a does not divide (x + u + v) for any distinct u,v in S\{a}
            # (x + u + v) % a == 0  <=>  (u + v) % a == (-x) % a
            if valid:
                for a in S:
                    target = (-x) % a
                    # R[a] contains residues (u+v)%a for u,v in S\{a}
                    if target in R[a]:
                        valid = False
                        break
            
            if valid:
                old_S = S[:]
                S.append(x)
                
                # Update K: Add new sums involving x
                # New sums are x + u + v where u, v are distinct elements from old_S
                n_old = len(old_S)
                for i in range(n_old):
                    u = old_S[i]
                    for j in range(i + 1, n_old):
                        v = old_S[j]
                        K.add(x + u + v)
                
                # Update R for existing elements in old_S
                for a in old_S:
                    # New pairs involving x: (x, u) where u in old_S, u != a
                    # Residue: (x + u) % a
                    for u in old_S:
                        if u != a:
                            R[a].add((x + u) % a)
                
                # Initialize R for new element x
                # Pairs in S\{x} are pairs in old_S
                R[x] = set()
                for i in range(n_old):
                    u = old_S[i]
                    for j in range(i + 1, n_old):
                        v = old_S[j]
                        R[x].add((u + v) % x)
            
            if time.time() > deadline:
                break
                
        return S

    def try_local_improve(S, time_limit):
        """
        Attempts to improve set S by swapping elements or removing and adding.
        Returns improved set if found, else S.
        """
        S = list(S)
        S_set = set(S)
        candidates = [x for x in range(1, 193) if x not in S_set]
        
        # Shuffle candidates for randomization
        random.shuffle(candidates)
        
        improved = False
        for x in candidates:
            if time.time() > time_limit:
                break
            
            # Try to add x directly
            # We need to re-check validity with the fast structures, but since we are 
            # potentially modifying S, it's safer to just use a slower check for the 
            # local search step or rebuild structures. 
            # Given the small size (<=48), a O(N^4) check for a single addition is 
            # C(48,3)*48 ~ 180k ops, which is fast enough for a few hundred tries.
            
            # However, we can use a slightly optimized check for "can x be added to S"
            # without maintaining global K/R, just checking x against S.
            
            # Check if x divides any sum of 3 in S
            ok = True
            if len(S) >= 3:
                for comb in combinations(S, 3):
                    if sum(comb) % x == 0:
                        ok = False
                        break
            
            # Check if any a in S divides x + u + v for u,v in S\{a}
            if ok:
                for i, a in enumerate(S):
                    others = [S[j] for j in range(len(S)) if j != i]
                    # We need to check if (x + u + v) % a == 0 for any distinct u,v in others
                    # This is equivalent to (u+v) % a == (-x) % a
                    target = (-x) % a
                    found = False
                    n_others = len(others)
                    for u_idx in range(n_others):
                        u = others[u_idx]
                        for v_idx in range(u_idx + 1, n_others):
                            v = others[v_idx]
                            if (u + v) % a == target:
                                found = True
                                break
                        if found:
                            break
                    if found:
                        ok = False
                        break
            
            if ok:
                S.append(x)
                S_set.add(x)
                improved = True
                # Continue trying to add more? Or restart local search?
                # Let's restart local search with the new set to be safe and thorough
                return try_local_improve(S, time_limit)
                
        # If no direct addition works, try swaps
        # Swap an element a in S with an element x not in S
        # This is complex. Let's stick to additions for now as greedy usually gets close.
        # But if we are stuck at < 48, we might need swaps.
        
        return S

    best_set = []
    best_size = 0
    
    # 1. Deterministic orders
    orders = [
        list(range(192, 0, -1)),
        list(range(1, 193)),
    ]
    
    # 2. Random orders
    random.seed(42)
    for _ in range(100):
        if time.time() > deadline:
            break
        perm = list(range(1, 193))
        random.shuffle(perm)
        orders.append(perm)
        
    for order in orders:
        if time.time() > deadline:
            break
            
        S = fast_greedy(order)
        
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            
        # If we haven't reached 48, try local improvement
        if best_size < 48:
            # Allocate some time for local improvement
            local_deadline = min(deadline, time.time() + 2.0)
            improved_S = try_local_improve(best_set, local_deadline)
            if len(improved_S) > best_size:
                best_size = len(improved_S)
                best_set = improved_S

    return best_set
# EVOLVE-BLOCK-END
