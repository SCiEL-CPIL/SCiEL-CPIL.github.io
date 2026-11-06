"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
from math import gcd

def construct():
    start_time = time.time()
    deadline = start_time + 50  # Safety margin before 60s limit
    
    N = 512
    # Precompute compatibility: can[i] is a bitset of j in [1..N] such that lcm(i,j) <= N
    # Using 0-indexed: vertex i-1 corresponds to number i
    # We'll use Python integers as bitsets for speed
    
    can = [0] * (N + 1)  # can[i] has bit (j-1) set if lcm(i,j) <= N
    
    # Precompute for each i, which j are compatible
    for i in range(1, N + 1):
        mask = 0
        for j in range(1, N + 1):
            g = gcd(i, j)
            lcm_val = i * j // g
            if lcm_val <= N:
                mask |= (1 << (j - 1))
        can[i] = mask
    
    # Greedy baseline: build a maximal clique
    def greedy_clique():
        clique = []
        candidates = set(range(1, N + 1))
        while candidates:
            # Pick vertex with max degree in remaining candidates
            best = None
            best_deg = -1
            for v in candidates:
                deg = bin(can[v] & 0)  # placeholder
                # Count bits in can[v] & candidate_mask
                # Actually let's just pick greedily by smallest number first for simplicity
                # or by some heuristic
                pass
            # Simple greedy: pick smallest number, add if compatible with all in clique
            for v in sorted(candidates):
                compatible = True
                for u in clique:
                    if not (can[v] >> (u - 1)) & 1:
                        compatible = False
                        break
                if compatible:
                    clique.append(v)
                    candidates.discard(v)
                    break
            else:
                break
        return clique
    
    # Better greedy: try multiple orderings
    def greedy_clique_v2():
        best_clique = []
        # Try picking in different orders
        for order in [list(range(1, N + 1)), list(range(N, 0, -1))]:
            clique = []
            for v in order:
                compatible = True
                for u in clique:
                    if not (can[v] >> (u - 1)) & 1:
                        compatible = False
                        break
                if compatible:
                    clique.append(v)
            if len(clique) > len(best_clique):
                best_clique = clique
        return best_clique
    
    baseline = greedy_clique_v2()
    best = set(baseline)
    
    # Local search: try to swap elements to improve
    def local_search(clique_set, time_limit):
        clique_list = sorted(clique_set)
        improved = True
        while improved and time.time() < time_limit:
            improved = False
            n = len(clique_list)
            # Try removing one element and adding a different one
            for i in range(n):
                if time.time() >= time_limit:
                    break
                removed = clique_list[i]
                remaining = [x for x in clique_list if x != removed]
                # Find candidates not in clique that are compatible with all remaining
                best_add = None
                best_score = -1
                for cand in range(1, N + 1):
                    if cand in clique_set:
                        continue
                    # Check if cand is compatible with all in remaining
                    compatible = True
                    for u in remaining:
                        if not (can[cand] >> (u - 1)) & 1:
                            compatible = False
                            break
                    if compatible:
                        # Score: prefer candidates that allow more future additions
                        # Simple: just take the first one found, or use degree heuristic
                        if best_add is None:
                            best_add = cand
                            # Try to find a better one (higher degree to remaining)
                            deg = sum(1 for u in remaining if (can[cand] >> (u - 1)) & 1)
                            best_score = deg
                            if deg == len(remaining):
                                break
                        else:
                            deg = sum(1 for u in remaining if (can[cand] >> (u - 1)) & 1)
                            if deg > best_score:
                                best_add = cand
                                best_score = deg
                                if deg == len(remaining):
                                    break
                if best_add is not None:
                    clique_set.discard(removed)
                    clique_set.add(best_add)
                    clique_list = sorted(clique_set)
                    improved = True
        return clique_set
    
    # Run local search
    search_deadline = deadline - 2
    best = local_search(best, search_deadline)
    
    # Convert to list and return
    result = sorted(best)
    return result
# EVOLVE-BLOCK-END
