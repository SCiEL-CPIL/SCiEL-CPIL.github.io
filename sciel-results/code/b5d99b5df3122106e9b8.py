import time
import random
from math import gcd

def construct():
    N = 128
    deadline = time.time() + 55.0  # Safety margin

    # Precompute GCD table
    gcd_table = [[0] * (N + 1) for _ in range(N + 1)]
    for i in range(1, N + 1):
        for j in range(i, N + 1):
            gcd_table[i][j] = gcd(i, j)
            gcd_table[j][i] = gcd_table[i][j]

    # Precompute bad triples as bitmasks
    # bad_mask[a][b] is a bitmask of c such that {a,b,c} is a bad triple
    bad_mask = [[0] * (N + 1) for _ in range(N + 1)]
    for a in range(1, N + 1):
        for b in range(a + 1, N + 1):
            gab = gcd_table[a][b]
            mask = 0
            for c in range(1, N + 1):
                if c == a or c == b:
                    continue
                if gcd_table[b][c] == gab and gcd_table[c][a] == gab:
                    mask |= (1 << c)
            bad_mask[a][b] = mask
            bad_mask[b][a] = mask

    def iter_bits(mask):
        while mask:
            lsb = mask & -mask
            yield lsb.bit_length() - 1
            mask ^= lsb

    def greedy_build(order):
        S = set()
        S_list = []
        conflict = [0] * (N + 1)
        
        for x in order:
            if conflict[x] == 0:
                S.add(x)
                S_list.append(x)
                # Update conflicts
                for a in S_list[:-1]:
                    mask = bad_mask[a][x]
                    for c in iter_bits(mask):
                        conflict[c] += 1
        
        return S, conflict

    def local_search(S, conflict, time_limit=5.0):
        S_list = list(S)
        start = time.time()
        
        while time.time() - start < time_limit:
            improved = False
            
            # 1. Try to add any element
            for x in range(1, N + 1):
                if x not in S and conflict[x] == 0:
                    S.add(x)
                    S_list.append(x)
                    for a in S_list[:-1]:
                        mask = bad_mask[a][x]
                        for c in iter_bits(mask):
                            conflict[c] += 1
                    improved = True
                    break
            
            # 2. Try swaps if no add
            if not improved:
                # Try to find a swap (u, v) such that v can be added to S \ {u}
                # and ideally allows more additions later
                # For now, just any valid swap that doesn't reduce size
                # Actually, we want to escape local optima.
                # A valid swap means conflict_{S\{u}}[v] == 0.
                # conflict_{S\{u}}[v] = conflict[v] - sum(a in S\{u}, v in bad_mask[a][u])
                
                found_swap = False
                for u in S_list:
                    # Calculate conflict if u is removed
                    # We need to check if any v can be added
                    # Let's just check all v
                    for v in range(1, N + 1):
                        if v in S:
                            continue
                        
                        # Calculate conflict of v in S \ {u}
                        conf_v = conflict[v]
                        # Subtract conflicts caused by u
                        # Conflicts caused by u are from pairs (a, u) where a in S\{u}
                        # and v in bad_mask[a][u]
                        # Note: bad_mask is symmetric
                        for a in S_list:
                            if a == u:
                                continue
                            if (bad_mask[a][u] >> v) & 1:
                                conf_v -= 1
                        
                        if conf_v == 0:
                            # Perform swap
                            S.remove(u)
                            S.add(v)
                            
                            # Update conflict array
                            # Remove u:
                            for a in S_list:
                                if a == u:
                                    continue
                                mask = bad_mask[a][u]
                                for c in iter_bits(mask):
                                    conflict[c] -= 1
                            
                            # Add v:
                            for a in S_list:
                                if a == v:
                                    continue
                                mask = bad_mask[a][v]
                                for c in iter_bits(mask):
                                    conflict[c] += 1
                            
                            S_list.remove(u)
                            S_list.append(v)
                            found_swap = True
                            improved = True
                            break
                    if found_swap:
                        break
            
            if not improved:
                break
        
        return S, conflict

    best_size = 0
    best_set = set()
    
    # Multiple restarts
    base_order = list(range(1, N + 1))
    
    # 1. Natural order
    S, conf = greedy_build(base_order)
    S, conf = local_search(S, conf, time_limit=2.0)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # 2. Reverse order
    S, conf = greedy_build(base_order[::-1])
    S, conf = local_search(S, conf, time_limit=2.0)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # 3. Sorted by number of divisors (ascending)
    div_count = [0] * (N + 1)
    for i in range(1, N + 1):
        for j in range(1, i + 1):
            if i % j == 0:
                div_count[i] += 1
    order_div = sorted(range(1, N + 1), key=lambda x: div_count[x])
    S, conf = greedy_build(order_div)
    S, conf = local_search(S, conf, time_limit=2.0)
    if len(S) > best_size:
        best_size = len(S)
        best_set = S

    # 4. Random restarts
    no_improve = 0
    while time.time() < deadline:
        order = base_order[:]
        random.shuffle(order)
        S, conf = greedy_build(order)
        S, conf = local_search(S, conf, time_limit=1.0)
        if len(S) > best_size:
            best_size = len(S)
            best_set = S
            no_improve = 0
        else:
            no_improve += 1
        if no_improve > 50:
            break

    # Final polish: try to add any remaining
    # Rebuild conflict for best_set
    conflict = [0] * (N + 1)
    S_list = list(best_set)
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            mask = bad_mask[a][b]
            for c in iter_bits(mask):
                conflict[c] += 1
                
    for x in range(1, N + 1):
        if x not in best_set and conflict[x] == 0:
            best_set.add(x)
            # Update conflict
            for a in S_list:
                mask = bad_mask[a][x]
                for c in iter_bits(mask):
                    conflict[c] += 1
            S_list.append(x)

    return list(best_set) if best_set else [1]
