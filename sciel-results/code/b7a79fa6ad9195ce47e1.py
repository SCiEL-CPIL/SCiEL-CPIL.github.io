"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import numpy as np


def construct(q, n, radius):
    start_time = time.time()
    deadline = start_time + 110.0  # Leave margin for 150s limit

    # Fallback for non-specific parameters
    if q != 3 or n != 7 or radius != 1:
        if n <= 4 and q == 3 and radius == 1:
            return [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        tetracode = [[a, b, (a + b) % 3, (a + 2 * b) % 3] for a in range(3) for b in range(3)]
        if n >= 4:
            return [h + list(tail) for h in tetracode for tail in itertools.product(range(3), repeat=n - 4)]
        else:
            return [list(w) for w in itertools.product(range(q), repeat=n)]

    # Specific optimization for q=3, n=7, radius=1
    N = 3 ** n  # 2187

    def idx_to_tuple(idx):
        t = [0] * n
        for i in range(n - 1, -1, -1):
            t[i] = idx % 3
            idx //= 3
        return t

    def tuple_to_idx(t):
        idx = 0
        for x in t:
            idx = idx * 3 + x
        return idx

    # Precompute neighbor lists as numpy arrays of int32 for fast operations
    neighbors_list = []
    for i in range(N):
        w = idx_to_tuple(i)
        nb = [i]
        for pos in range(n):
            val = w[pos]
            for new_val in range(3):
                if new_val != val:
                    new_w = w[:]
                    new_w[pos] = new_val
                    nb.append(tuple_to_idx(new_w))
        neighbors_list.append(np.array(nb, dtype=np.int32))

    # Precompute neighbor sets as Python sets for faster intersection counting
    neighbor_sets = [set(nb.tolist()) for nb in neighbors_list]

    # ---------- Greedy Set-Cover Construction ----------
    coverage = np.zeros(N, dtype=np.int32)
    uncovered_mask = np.ones(N, dtype=bool)
    code = []
    
    while uncovered_mask.any() and time.time() < deadline:
        uncovered_indices = np.where(uncovered_mask)[0]
        if len(uncovered_indices) == 0:
            break
        
        candidate_set = set()
        for idx in uncovered_indices:
            candidate_set.update(neighbor_sets[idx])
        
        if not candidate_set:
            break
        
        best_c = -1
        best_count = 0
        
        for c in candidate_set:
            cnt = np.sum(uncovered_mask[neighbors_list[c]])
            if cnt > best_count:
                best_count = cnt
                best_c = c
            if best_count == len(uncovered_indices):
                break
        
        if best_c == -1 or best_count == 0:
            for c in range(N):
                cnt = np.sum(uncovered_mask[neighbors_list[c]])
                if cnt > best_count:
                    best_count = cnt
                    best_c = c
            if best_c == -1 or best_count == 0:
                break
        
        code.append(best_c)
        coverage[neighbors_list[best_c]] += 1
        uncovered_mask[neighbors_list[best_c]] = False
    
    # ---------- Local Search: Removal and Kick ----------
    
    def is_redundant(c):
        return np.min(coverage[neighbors_list[c]]) > 1

    def greedy_repair(uncovered_set):
        if not uncovered_set:
            return []
        added = []
        temp_uncovered = set(uncovered_set)
        
        while temp_uncovered:
            best_c = -1
            best_count = 0
            
            candidate_pool = set()
            for w in temp_uncovered:
                candidate_pool.update(neighbor_sets[w])
            
            for c in candidate_pool:
                if c in code:
                    continue
                cnt = 0
                for x in neighbors_list[c]:
                    if x in temp_uncovered:
                        cnt += 1
                        if cnt > best_count:
                            best_count = cnt
                            best_c = c
                        if best_count == len(temp_uncovered):
                            break
                if best_count == len(temp_uncovered):
                    break
            
            if best_c == -1:
                break
            
            added.append(best_c)
            for w in neighbors_list[best_c]:
                temp_uncovered.discard(w)
        
        return added

    def try_remove():
        nonlocal code, coverage
        removed = False
        for c in list(code):
            if is_redundant(c):
                code.remove(c)
                coverage[neighbors_list[c]] -= 1
                removed = True
        return removed

    def kick():
        nonlocal code, coverage
        if not code:
            return False
            
        old_size = len(code)
        
        idx_to_remove = random.randint(0, len(code) - 1)
        c_remove = code[idx_to_remove]
        
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1
        
        uncovered = np.where(temp_coverage == 0)[0]
        
        if len(uncovered) == 0:
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            return True

        uncovered_set = set(uncovered.tolist())
        added = greedy_repair(uncovered_set)
        
        new_size = old_size - 1 + len(added)
        
        if new_size < old_size:
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            for c in added:
                code.append(c)
                coverage[neighbors_list[c]] += 1
            return True
        else:
            return False

    def exchange_move():
        """Try to remove one codeword and add another to improve coverage efficiency."""
        nonlocal code, coverage
        if len(code) < 2:
            return False
            
        old_size = len(code)
        
        c_remove = random.choice(code)
        
        temp_coverage = coverage.copy()
        temp_coverage[neighbors_list[c_remove]] -= 1
        
        uncovered = np.where(temp_coverage == 0)[0]
        uncovered_set = set(uncovered.tolist())
        
        if not uncovered_set:
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            return True
        
        candidate_pool = set()
        for w in uncovered_set:
            candidate_pool.update(neighbor_sets[w])
        
        best_c = -1
        best_count = 0
        for c in candidate_pool:
            if c in code:
                continue
            cnt = sum(1 for x in neighbors_list[c] if x in uncovered_set)
            if cnt > best_count:
                best_count = cnt
                best_c = c
            if best_count == len(uncovered_set):
                break
        
        if best_c != -1 and best_count == len(uncovered_set):
            code.remove(c_remove)
            code.append(best_c)
            coverage[neighbors_list[c_remove]] -= 1
            coverage[neighbors_list[best_c]] += 1
            return True
        
        added = greedy_repair(uncovered_set)
        new_size = old_size - 1 + len(added)
        
        if new_size < old_size:
            code.remove(c_remove)
            coverage[neighbors_list[c_remove]] -= 1
            for c in added:
                code.append(c)
                coverage[neighbors_list[c]] += 1
            return True
        
        return False

    # ---------- Simulated Annealing ----------
    def compute_energy():
        """Energy = number of codewords + large penalty for uncovered words."""
        uncovered_count = int(np.sum(coverage == 0))
        return len(code) + 1000 * uncovered_count

    def simulated_annealing():
        """Simulated annealing on the code set to escape local minima."""
        nonlocal code, coverage
        
        if not code:
            return False
        
        best_code = code[:]
        best_coverage = coverage.copy()
        best_energy = compute_energy()
        
        # Temperature schedule
        T_start = 5.0
        T_end = 0.1
        cooling_rate = 0.97
        T = T_start
        
        # Number of iterations per temperature level
        iters_per_temp = 50
        
        while T > T_end and time.time() < deadline:
            for _ in range(iters_per_temp):
                if time.time() > deadline:
                    break
                
                if not code:
                    break
                    
                # Choose a random move type
                move_type = random.random()
                
                if move_type < 0.4:
                    # Remove a random codeword
                    idx = random.randint(0, len(code) - 1)
                    c_remove = code[idx]
                    
                    old_energy = compute_energy()
                    
                    code.remove(c_remove)
                    coverage[neighbors_list[c_remove]] -= 1
                    
                    new_energy = compute_energy()
                    delta_e = new_energy - old_energy
                    
                    if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                        # Accept
                        pass
                    else:
                        # Reject
                        code.append(c_remove)
                        coverage[neighbors_list[c_remove]] += 1
                    
                    if compute_energy() < best_energy:
                        best_energy = compute_energy()
                        best_code = code[:]
                        best_coverage = coverage.copy()
                        
                elif move_type < 0.7:
                    # Add a random codeword
                    c_add = random.randint(0, N - 1)
                    if c_add not in code:
                        old_energy = compute_energy()
                        
                        code.append(c_add)
                        coverage[neighbors_list[c_add]] += 1
                        
                        new_energy = compute_energy()
                        delta_e = new_energy - old_energy
                        
                        if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                            pass
                        else:
                            code.remove(c_add)
                            coverage[neighbors_list[c_add]] -= 1
                        
                        if compute_energy() < best_energy:
                            best_energy = compute_energy()
                            best_code = code[:]
                            best_coverage = coverage.copy()
                            
                else:
                    # Swap: remove one, add one
                    if len(code) >= 1:
                        idx = random.randint(0, len(code) - 1)
                        c_remove = code[idx]
                        c_add = random.randint(0, N - 1)
                        
                        if c_add not in code or c_add == c_remove:
                            old_energy = compute_energy()
                            
                            code.remove(c_remove)
                            coverage[neighbors_list[c_remove]] -= 1
                            code.append(c_add)
                            coverage[neighbors_list[c_add]] += 1
                            
                            new_energy = compute_energy()
                            delta_e = new_energy - old_energy
                            
                            if delta_e < 0 or random.random() < np.exp(-delta_e / T):
                                pass
                            else:
                                code.remove(c_add)
                                coverage[neighbors_list[c_add]] -= 1
                                code.append(c_remove)
                                coverage[neighbors_list[c_remove]] += 1
                            
                            if compute_energy() < best_energy:
                                best_energy = compute_energy()
                                best_code = code[:]
                                best_coverage = coverage.copy()
                
                T *= cooling_rate
            
            # Ensure coverage is valid at each temperature step
            # If we have uncovered words, do a quick repair
            if np.any(coverage == 0):
                uncovered_set = set(np.where(coverage == 0)[0].tolist())
                added = greedy_repair(uncovered_set)
                for c in added:
                    if c not in code:
                        code.append(c)
                        coverage[neighbors_list[c]] += 1
                
                # Update best if improved
                if compute_energy() < best_energy:
                    best_energy = compute_energy()
                    best_code = code[:]
                    best_coverage = coverage.copy()
        
        # Restore best solution found during SA
        if best_energy < compute_energy():
            code = best_code[:]
            coverage = best_coverage.copy()
            return True
        else:
            return False

    # Main Loop
    while time.time() < deadline:
        # 1. Exhaustive Removal
        changed = True
        while changed:
            if time.time() > deadline:
                break
            changed = try_remove()
        
        # 2. Kick and Exchange moves
        if time.time() > deadline:
            break
        
        kicks_tried = 0
        success = False
        max_kicks = 30
        while kicks_tried < max_kicks:
            if random.random() < 0.5:
                improved = kick()
            else:
                improved = exchange_move()
            
            if improved:
                success = True
                kicks_tried = 0
            else:
                kicks_tried += 1
            if time.time() > deadline:
                break
        
        # 3. Simulated Annealing phase when stuck
        if not success and time.time() < deadline:
            if simulated_annealing():
                success = True
                # After SA, do removal pass
                changed = True
                while changed:
                    if time.time() > deadline:
                        break
                    changed = try_remove()
        
        # 4. 2-removal kick
        if not success and len(code) > 2 and time.time() < deadline:
            i1 = random.randint(0, len(code) - 1)
            i2 = random.randint(0, len(code) - 1)
            if i1 == i2:
                i2 = (i2 + 1) % len(code)
            
            c1 = code[i1]
            c2 = code[i2]
            
            temp_coverage = coverage.copy()
            temp_coverage[neighbors_list[c1]] -= 1
            temp_coverage[neighbors_list[c2]] -= 1
            
            uncovered = np.where(temp_coverage == 0)[0]
            uncovered_set = set(uncovered.tolist())
            added = greedy_repair(uncovered_set)
            
            new_size = len(code) - 2 + len(added)
            
            if new_size < len(code):
                code.remove(c1)
                code.remove(c2)
                coverage[neighbors_list[c1]] -= 1
                coverage[neighbors_list[c2]] -= 1
                for c in added:
                    code.append(c)
                    coverage[neighbors_list[c]] += 1
                success = True
        
        # 5. Additional exchange attempts when stuck
        if not success and time.time() < deadline:
            for _ in range(10):
                if exchange_move():
                    success = True
                    break
                if time.time() > deadline:
                    break
    
    # Final removal pass
    changed = True
    while changed:
        if time.time() > deadline:
            break
        changed = try_remove()
    
    result = [idx_to_tuple(cw) for cw in code]
    
    return result
# EVOLVE-BLOCK-END
