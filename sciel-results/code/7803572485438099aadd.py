import numpy as np
import sys
import time
import random
from itertools import combinations

def furstenberg_sarkozy_205() -> np.ndarray:
    m = 205
    start_time = time.time()
    time_limit = 580  # seconds, leave margin for 630s total

    # Forbidden differences mod 205
    forbidden_205 = set()
    for x in range(1, m):
        forbidden_205.add(pow(x, 2, m))
    forbidden_205.discard(0)

    # Squares mod 41 and mod 5
    squares_41 = set()
    for x in range(1, 41):
        squares_41.add(pow(x, 2, 41))
    squares_41.discard(0)

    squares_5 = set()
    for x in range(1, 5):
        squares_5.add(pow(x, 2, 5))
    squares_5.discard(0)
    # squares_5 = {1, 4} mod 5

    # Non-squares mod 5
    nonsquares_5 = set(range(5)) - squares_5
    # nonsquares_5 = {2, 3} mod 5

    # Build the 5x41 grid
    # Each cell (a, b) corresponds to residue x mod 205 where x ≡ a (mod 5) and x ≡ b (mod 41)
    # By CRT, this is a bijection between {0..4} x {0..40} and {0..204}
    
    # Precompute the CRT mapping
    # x ≡ a (mod 5), x ≡ b (mod 41)
    # x = a * 41 * inv41_mod5 + b * 5 * inv5_mod41 (mod 205)
    # 41 mod 5 = 1, so inv41_mod5 = 1
    # 5 mod 41 = 5, inv5_mod41 = 33 (since 5*33 = 165 = 4*41 + 1)
    
    def crt(a, b):
        return (a * 41 + b * 165) % 205
    
    # For each pair of cells (a1,b1) and (a2,b2), check if they form a forbidden difference
    # Difference d = (crt(a1,b1) - crt(a2,b2)) mod 205
    # d mod 5 = (a1 - a2) mod 5
    # d mod 41 = (b1 - b2) mod 41
    # d is in forbidden_205 iff d is a nonzero square mod 205
    # By CRT, d is a square mod 205 iff d mod 5 is a square mod 5 AND d mod 41 is a square mod 41
    # (including 0 as a square, but we discard 0 from forbidden)
    
    # Actually, let me reconsider. d is a square mod 205 iff there exists x such that x^2 ≡ d (mod 205).
    # By CRT, this means x^2 ≡ d (mod 5) and x^2 ≡ d (mod 41).
    # So d mod 5 must be a square mod 5 (including 0) and d mod 41 must be a square mod 41 (including 0).
    # But d ≠ 0 mod 205, so we need d mod 5 and d mod 41 to both be squares (possibly 0), but not both 0.
    
    # Let me define:
    # d is "forbidden" iff d mod 5 ∈ squares_5 ∪ {0} and d mod 41 ∈ squares_41 ∪ {0}, and d ≠ 0 mod 205.
    # Since d ≠ 0 mod 205 means (d mod 5, d mod 41) ≠ (0, 0).
    
    squares_5_with_0 = squares_5 | {0}
    squares_41_with_0 = squares_41 | {0}
    
    # Build adjacency on the 5x41 grid
    # Two cells (a1,b1) and (a2,b2) are adjacent (conflict) if their difference is forbidden
    # i.e., (a1-a2) mod 5 ∈ squares_5_with_0 and (b1-b2) mod 41 ∈ squares_41_with_0, and not both 0.
    
    n_a = 5
    n_b = 41
    
    # Represent each cell as index = a * 41 + b
    total_cells = n_a * n_b
    
    # Build conflict graph
    # conflict[i][j] = True if cells i and j cannot both be in the set
    conflict = [[False] * total_cells for _ in range(total_cells)]
    
    for a1 in range(n_a):
        for b1 in range(n_b):
            i = a1 * n_b + b1
            for a2 in range(n_a):
                for b2 in range(n_b):
                    if i == a2 * n_b + b2:
                        continue
                    j = a2 * n_b + b2
                    da = (a1 - a2) % 5
                    db = (b1 - b2) % 41
                    if da in squares_5_with_0 and db in squares_41_with_0:
                        if da != 0 or db != 0:
                            conflict[i][j] = True
                            conflict[j][i] = True
    
    print(f"Conflict graph built. Total cells: {total_cells}", file=sys.stderr)
    
    # Now find maximum independent set on this 205-node graph
    # Use branch and bound with good heuristics
    
    # Convert to adjacency list
    adj_list = [set() for _ in range(total_cells)]
    for i in range(total_cells):
        for j in range(total_cells):
            if conflict[i][j]:
                adj_list[i].add(j)
    
    # Degree of each node
    degree = [len(adj_list[i]) for i in range(total_cells)]
    
    # Use a recursive branch and bound
    # State: current set, candidates that can still be added
    # We'll use a bitset for efficiency
    
    # Actually, let's use a simpler approach: backtracking with pruning
    # Order nodes by degree (highest first) for better pruning
    
    # Precompute for each node, the set of nodes it conflicts with (as a bitmask)
    # 205 nodes, so we can use Python integers as bitmasks
    
    bitmask_conflict = [0] * total_cells
    for i in range(total_cells):
        for j in adj_list[i]:
            bitmask_conflict[i] |= (1 << j)
    
    # Branch and bound
    best_size = 0
    best_set = []
    
    def bnb(candidates_mask, current_set, depth):
        nonlocal best_size, best_set
        
        if time.time() - start_time > time_limit:
            return
        
        # Upper bound: current size + number of candidates
        if len(current_set) + bin(candidates_mask).count('1') <= best_size:
            return
        
        if candidates_mask == 0:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Choose a node to branch on
        # Heuristic: pick the node with highest degree among candidates
        # Extract candidate nodes
        cand_list = []
        temp = candidates_mask
        while temp:
            lsb = temp & (-temp)
            idx = lsb.bit_length() - 1
            cand_list.append(idx)
            temp ^= lsb
        
        if not cand_list:
            if len(current_set) > best_size:
                best_size = len(current_set)
                best_set = list(current_set)
            return
        
        # Pick node with max degree within candidates
        # For each candidate, count how many other candidates it conflicts with
        best_node = cand_list[0]
        best_deg = -1
        for v in cand_list:
            # Count conflicts with other candidates
            cnt = 0
            for u in cand_list:
                if u != v and (bitmask_conflict[v] >> u) & 1:
                    cnt += 1
            if cnt > best_deg:
                best_deg = cnt
                best_node = v
        
        v = best_node
        
        # Branch 1: include v
        new_candidates = candidates_mask & ~(1 << v)
        # Remove all nodes that conflict with v
        new_candidates &= ~bitmask_conflict[v]
        current_set.append(v)
        bnb(new_candidates, current_set, depth + 1)
        current_set.pop()
        
        # Branch 2: exclude v
        new_candidates = candidates_mask & ~(1 << v)
        bnb(new_candidates, current_set, depth + 1)
    
    # Initial candidates: all nodes
    all_mask = (1 << total_cells) - 1
    bnb(all_mask, [], 0)
    
    print(f"Branch and bound found set of size {best_size}", file=sys.stderr)
    
    # If BnB didn't find anything good, try greedy as fallback
    if best_size == 0:
        # Greedy
        current = set()
        remaining = set(range(total_cells))
        while remaining:
            # Pick node with min conflicts with current
            best_v = None
            best_conf = float('inf')
            for v in remaining:
                conf = 0
                for u in current:
                    if conflict[v][u]:
                        conf += 1
                if conf < best_conf:
                    best_conf = conf
                    best_v = v
            if best_v is not None:
                current.add(best_v)
                remaining.discard(best_v)
        best_size = len(current)
        best_set = list(current)
        print(f"Greedy found set of size {best_size}", file=sys.stderr)
    
    # Convert back to residues mod 205
    result = set()
    for idx in best_set:
        a = idx // n_b
        b = idx % n_b
        x = crt(a, b)
        result.add(x)
    
    result_arr = np.array(sorted(result), dtype=np.int64)
    
    # Verify
    valid = True
    for i in range(len(result_arr)):
        for j in range(i+1, len(result_arr)):
            d1 = (result_arr[i] - result_arr[j]) % 205
            d2 = (result_arr[j] - result_arr[i]) % 205
            if d1 in forbidden_205 or d2 in forbidden_205:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        print("VALIDATION FAILED, using trivial set", file=sys.stderr)
        return np.array([0], dtype=np.int64)
    
    print(f"Final result size: {len(result_arr)}", file=sys.stderr)
    return result_arr
