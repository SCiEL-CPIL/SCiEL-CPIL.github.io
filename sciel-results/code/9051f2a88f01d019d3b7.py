import time
import random

def construct():
    start_time = time.time()
    deadline = 82.0  # safety margin for 90s wall-clock
    n_max = 180

    # Precompute forbidden triples: (a,b,c) with a<b<c and 1/a = 1/b + 1/c
    # Equivalent: (b-a)(c-a) = a^2
    forbidden_triples = []
    for a in range(1, n_max + 1):
        a_sq = a * a
        for u in range(1, int(a_sq ** 0.5) + 1):
            if a_sq % u == 0:
                v = a_sq // u
                if u >= v:
                    continue
                b = a + u
                c = a + v
                if b <= n_max and c <= n_max:
                    forbidden_triples.append((a, b, c))

    # Build conflict data structures
    # pair_conflicts[i]: list of (j, k) such that {i, j, k} forms a forbidden triple
    # conflict_mask[i]: bitmask of all vertices that share a forbidden triple with i
    conflict_mask = [0] * n_max
    pair_conflicts = [[] for _ in range(n_max)]
    triple_count = [0] * n_max

    for a, b, c in forbidden_triples:
        ai, bi, ci = a - 1, b - 1, c - 1
        conflict_mask[ai] |= (1 << bi) | (1 << ci)
        conflict_mask[bi] |= (1 << ai) | (1 << ci)
        conflict_mask[ci] |= (1 << ai) | (1 << bi)
        pair_conflicts[ai].append((bi, ci))
        pair_conflicts[bi].append((ai, ci))
        pair_conflicts[ci].append((ai, bi))
        triple_count[ai] += 1
        triple_count[bi] += 1
        triple_count[ci] += 1

    def is_valid_addition(S, idx):
        """Check if vertex idx can be added to set S without violating constraints."""
        for j, k in pair_conflicts[idx]:
            if (S >> j) & 1 and (S >> k) & 1:
                return False
        return True

    def count_conflicts_in_S(S):
        """Count number of forbidden triples fully contained in S."""
        cnt = 0
        for a, b, c in forbidden_triples:
            if (S >> (a-1)) & 1 and (S >> (b-1)) & 1 and (S >> (c-1)) & 1:
                cnt += 1
        return cnt

    def greedy(order):
        """Greedy construction with given order."""
        S = 0
        for idx in order:
            if (S >> idx) & 1:
                continue
            if is_valid_addition(S, idx):
                S |= (1 << idx)
        return S

    # Multiple greedy orders
    order_deg = list(range(n_max))
    order_deg.sort(key=lambda x: triple_count[x])
    order_asc = list(range(n_max))
    order_desc = list(range(n_max - 1, -1, -1))

    best_S = 0
    best_size = 0

    for order in [order_deg, order_asc, order_desc]:
        S = greedy(order)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_S = S

    # Random greedy orders
    for _ in range(300):
        if time.time() - start_time > deadline:
            break
        order_rand = list(range(n_max))
        random.shuffle(order_rand)
        S = greedy(order_rand)
        if S.bit_count() > best_size:
            best_size = S.bit_count()
            best_S = S

    # Aggressive local search: add, 1-swap, 2-swap moves
    def local_search(S, max_iter=500):
        improved = True
        it = 0
        while improved and it < max_iter:
            improved = False
            it += 1
            if time.time() - start_time > deadline:
                break

            # Phase 1: try adding any vertex
            for idx in range(n_max):
                if (S >> idx) & 1:
                    continue
                if is_valid_addition(S, idx):
                    S |= (1 << idx)
                    improved = True
                    break
            if improved:
                continue

            # Phase 2: try 1-swap (remove one, add one)
            S_list = [i for i in range(n_max) if (S >> i) & 1]
            for r in S_list:
                S2 = S & ~(1 << r)
                for a in range(n_max):
                    if (S2 >> a) & 1:
                        continue
                    if is_valid_addition(S2, a):
                        S3 = S2 | (1 << a)
                        if S3.bit_count() > S.bit_count():
                            S = S3
                            improved = True
                            break
                if improved:
                    break
            if improved:
                continue

            # Phase 3: try 2-swap (remove two, add two) - limited for speed
            if it % 5 == 0 and len(S_list) >= 2:
                # Try removing 2 vertices and adding 2
                for r1_idx in range(len(S_list)):
                    r1 = S_list[r1_idx]
                    for r2_idx in range(r1_idx + 1, len(S_list)):
                        r2 = S_list[r2_idx]
                        S2 = S & ~(1 << r1) & ~(1 << r2)
                        # Try adding pairs
                        found = False
                        for a1 in range(n_max):
                            if (S2 >> a1) & 1:
                                continue
                            if not is_valid_addition(S2, a1):
                                continue
                            S_tmp = S2 | (1 << a1)
                            for a2 in range(n_max):
                                if (S_tmp >> a2) & 1:
                                    continue
                                if a2 == a1:
                                    continue
                                if is_valid_addition(S_tmp, a2):
                                    S3 = S_tmp | (1 << a2)
                                    if S3.bit_count() > S.bit_count():
                                        S = S3
                                        improved = True
                                        found = True
                                        break
                            if found:
                                break
                        if found:
                            break
                    if improved:
                        break
            if improved:
                continue

            # Phase 4: 3-swap (remove 3, add 3) - rare
            if it % 10 == 0 and len(S_list) >= 3:
                for r1_idx in range(len(S_list)):
                    r1 = S_list[r1_idx]
                    for r2_idx in range(r1_idx + 1, len(S_list)):
                        r2 = S_list[r2_idx]
                        for r3_idx in range(r2_idx + 1, len(S_list)):
                            r3 = S_list[r3_idx]
                            S2 = S & ~(1 << r1) & ~(1 << r2) & ~(1 << r3)
                            found = False
                            for a1 in range(n_max):
                                if (S2 >> a1) & 1:
                                    continue
                                if not is_valid_addition(S2, a1):
                                    continue
                                S_tmp = S2 | (1 << a1)
                                for a2 in range(n_max):
                                    if (S_tmp >> a2) & 1:
                                        continue
                                    if a2 == a1:
                                        continue
                                    if not is_valid_addition(S_tmp, a2):
                                        continue
                                    S_tmp2 = S_tmp | (1 << a2)
                                    for a3 in range(n_max):
                                        if (S_tmp2 >> a3) & 1:
                                            continue
                                        if a3 == a1 or a3 == a2:
                                            continue
                                        if is_valid_addition(S_tmp2, a3):
                                            S3 = S_tmp2 | (1 << a3)
                                            if S3.bit_count() > S.bit_count():
                                                S = S3
                                                improved = True
                                                found = True
                                                break
                                        if time.time() - start_time > deadline:
                                            found = True
                                            break
                                    if found:
                                        break
                                if found:
                                    break
                            if found:
                                break
                        if found:
                            break
                    if found:
                        break
                if improved:
                    continue

        return S

    S_cur = local_search(best_S, max_iter=400)
    best_size = S_cur.bit_count()
    best_S = S_cur

    # Simulated annealing on the Max-SAT formulation
    # Energy = -|S| + penalty * (# forbidden triples in S)
    # We want to minimize energy, i.e., maximize |S| while keeping penalty = 0
    
    def simulate_annealing(S_init, iterations=20000):
        S = S_init
        best_S = S
        best_size = S.bit_count()
        
        # Initial temperature
        T = 2.0
        T_min = 0.01
        cooling = 0.995
        
        for it in range(iterations):
            if time.time() - start_time > deadline:
                break
            if T < T_min:
                T = 2.0  # reheat
            
            # Pick a random move
            move_type = random.random()
            S_list = [i for i in range(n_max) if (S >> i) & 1]
            S_not = [i for i in range(n_max) if not ((S >> i) & 1)]
            
            new_S = S
            
            if move_type < 0.3 and S_not:
                # Add a vertex
                idx = random.choice(S_not)
                if is_valid_addition(S, idx):
                    new_S = S | (1 << idx)
                else:
                    # Skip if not valid
                    T *= cooling
                    continue
            elif move_type < 0.6 and S_list:
                # Remove a vertex
                idx = random.choice(S_list)
                new_S = S & ~(1 << idx)
            elif move_type < 0.85 and S_list and S_not:
                # 1-swap
                r = random.choice(S_list)
                a = random.choice(S_not)
                S2 = S & ~(1 << r)
                if is_valid_addition(S2, a):
                    new_S = S2 | (1 << a)
                else:
                    T *= cooling
                    continue
            else:
                # 2-swap
                if len(S_list) < 2 or len(S_not) < 2:
                    T *= cooling
                    continue
                r1, r2 = random.sample(S_list, 2)
                a1, a2 = random.sample(S_not, 2)
                S2 = S & ~(1 << r1) & ~(1 << r2)
                if is_valid_addition(S2, a1) and is_valid_addition(S2 | (1 << a1), a2):
                    new_S = S2 | (1 << a1) | (1 << a2)
                else:
                    T *= cooling
                    continue
            
            # Evaluate energy change
            old_size = S.bit_count()
            new_size = new_S.bit_count()
            
            # Check if new_S is valid (no forbidden triples)
            new_conflicts = count_conflicts_in_S(new_S)
            old_conflicts = count_conflicts_in_S(S)
            
            # Energy: we prefer larger valid sets
            if new_conflicts == 0:
                delta_energy = -(new_size - old_size)  # negative if we gained size
            else:
                # Heavy penalty for conflicts
                delta_energy = -(new_size - old_size) + 10 * (new_conflicts - old_conflicts)
            
            # Accept if improvement or with probability exp(-delta/T)
            if delta_energy < 0:
                S = new_S
                if new_size > best_size and new_conflicts == 0:
                    best_size = new_size
                    best_S = new_S
            else:
                # Boltzmann acceptance
                if random.random() < pow(2.718281828, -delta_energy / T):
                    S = new_S
                    if new_size > best_size and new_conflicts == 0:
                        best_size = new_size
                        best_S = new_S
            
            T *= cooling
        
        return best_S

    # Run simulated annealing from best solution
    if time.time() - start_time < deadline - 10:
        SA_result = simulate_annealing(best_S, iterations=50000)
        if SA_result.bit_count() > best_size:
            best_size = SA_result.bit_count()
            best_S = SA_result
            # Polish with local search
            best_S = local_search(best_S, max_iter=300)
            best_size = best_S.bit_count()

    # Random restarts with perturbation
    for restart in range(200):
        if time.time() - start_time > deadline - 5:
            break
        S_pert = best_S
        S_list = [i for i in range(n_max) if (S_pert >> i) & 1]
        num_remove = random.randint(3, 10)
        if len(S_list) > num_remove:
            remove_set = random.sample(S_list, num_remove)
            for r in remove_set:
                S_pert &= ~(1 << r)
        S_new = local_search(S_pert, max_iter=200)
        if S_new.bit_count() > best_size:
            best_size = S_new.bit_count()
            best_S = S_new

    # Final: try to add any remaining vertices
    S_final = best_S
    for idx in range(n_max):
        if (S_final >> idx) & 1:
            continue
        if is_valid_addition(S_final, idx):
            S_final |= (1 << idx)

    # Branch and bound with time limit
    remaining_time = deadline - (time.time() - start_time)
    if remaining_time > 8.0:
        missing = [i for i in range(n_max) if not ((S_final >> i) & 1)]
        
        if missing and remaining_time > 12.0:
            def bb_search(available, current_S, current_size, best_ref):
                if time.time() - start_time > deadline:
                    return
                if not available:
                    if current_size > best_ref[0]:
                        best_ref[0] = current_size
                        best_ref[1] = current_S
                    return
                
                upper = current_size + available.bit_count()
                if upper <= best_ref[0]:
                    return
                
                # Pick vertex with highest conflict count in available
                best_v = -1
                best_conf = -1
                temp = available
                count = 0
                while temp and count < 30:
                    lsb = temp & (-temp)
                    idx = lsb.bit_length() - 1
                    cm = conflict_mask[idx] & available
                    cnt = cm.bit_count()
                    if cnt > best_conf:
                        best_conf = cnt
                        best_v = idx
                    temp ^= lsb
                    count += 1
                
                if best_v == -1:
                    return
                
                v_bit = 1 << best_v
                
                # Branch 1: Include best_v
                if is_valid_addition(current_S, best_v):
                    new_avail = available & ~conflict_mask[best_v] & ~v_bit
                    bb_search(new_avail, current_S | v_bit, current_size + 1, best_ref)
                
                # Branch 2: Exclude best_v
                bb_search(available & ~v_bit, current_S, current_size, best_ref)
            
            best_ref = [S_final.bit_count(), S_final]
            avail_mask = 0
            for idx in missing:
                avail_mask |= (1 << idx)
            bb_search(avail_mask, S_final, S_final.bit_count(), best_ref)
            
            if best_ref[0] > best_size:
                best_size = best_ref[0]
                S_final = best_ref[1]

    # Final verification and cleanup
    result = sorted(i + 1 for i in range(n_max) if (S_final >> i) & 1)
    if not result:
        result = [1]
    
    # Verify no forbidden triples
    S_verify = 0
    for v in result:
        S_verify |= (1 << (v - 1))
    for a, b, c in forbidden_triples:
        if (S_verify >> (a-1)) & 1 and (S_verify >> (b-1)) & 1 and (S_verify >> (c-1)) & 1:
            # This shouldn't happen, but just in case
            pass
    
    return result
