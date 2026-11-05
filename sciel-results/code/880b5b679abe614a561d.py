import time
import random
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 85  # Safety margin
    
    n = 128
    target_min = 128
    target_max = 256
    num_sums = target_max + 1  # 257
    
    # Precompute delta vectors for self-pairs
    # delta_self[x] is a vector of length 257 with 1 at 2x if 2x in range, else 0
    delta_self = np.zeros((129, num_sums), dtype=np.int32)
    for x in range(129):
        s = 2 * x
        if target_min <= s <= target_max:
            delta_self[x, s] = 1

    def calculate_V(A_mask):
        """Calculate V matrix: V[x][s] = count of y in A such that x+y=s."""
        V = np.zeros((129, num_sums), dtype=np.int32)
        A_indices = np.where(A_mask)[0]
        for x in range(129):
            if len(A_indices) == 0:
                continue
            sums = x + A_indices
            valid = (sums >= 0) & (sums < num_sums)
            if np.any(valid):
                # Use bincount for speed
                # V[x, sums[valid]] += 1
                # But we need to accumulate.
                # Since A_indices is small, loop is fine or use add.at
                np.add.at(V[x], sums[valid], 1)
        return V

    def calculate_N(A_mask):
        """Calculate N vector: N[s] = count of unordered pairs (a,b) in A with a+b=s."""
        N = np.zeros(num_sums, dtype=np.int32)
        A_indices = np.where(A_mask)[0]
        m = len(A_indices)
        if m == 0:
            return N
        i_idx, j_idx = np.triu_indices(m)
        sums = A_indices[i_idx] + A_indices[j_idx]
        valid = (sums >= 0) & (sums < num_sums)
        if np.any(valid):
            np.add.at(N, sums[valid], 1)
        return N

    def get_objective(N):
        target_N = N[target_min:target_max+1]
        missing = np.sum(target_N == 0)
        if missing > 0:
            return (missing, float('inf'), float('inf'))
        peak = int(np.max(target_N))
        sum_sq = int(np.sum(target_N**2))
        return (0, peak, sum_sq)

    def greedy_init(k):
        A_mask = np.zeros(129, dtype=bool)
        # Start with a random set
        init_set = set(random.sample(range(129), k))
        for x in init_set:
            A_mask[x] = True
        
        N = calculate_N(A_mask)
        
        # Repair: add elements to cover missing sums
        # Simple greedy: add element that covers most missing sums
        while True:
            missing_sums = [s for s in range(target_min, target_max+1) if N[s] == 0]
            if not missing_sums:
                break
            
            best_x = None
            best_cover = -1
            candidates = [x for x in range(129) if not A_mask[x]]
            
            A_indices = np.where(A_mask)[0]
            
            for x in candidates:
                # Count how many missing sums are covered by adding x
                # New sums: x + y for y in A, and 2x
                covered = 0
                for s in missing_sums:
                    # Check if x + y = s for some y in A
                    y = s - x
                    if 0 <= y <= 128 and A_mask[y]:
                        covered += 1
                    # Check if 2x = s
                    if 2*x == s:
                        covered += 1
                if covered > best_cover:
                    best_cover = covered
                    best_x = x
            
            if best_x is None or best_cover == 0:
                break
            
            A_mask[best_x] = True
            # Update N
            A_indices = np.where(A_mask)[0]
            # Recalculate N is slow, but repair is short.
            # Or incremental update.
            # Let's just recalculate for simplicity in init, it's small.
            N = calculate_N(A_mask)
            
        return A_mask, N

    def local_search(A_mask, N, V, time_limit):
        """Best improvement local search with swap moves."""
        A_mask = A_mask.copy()
        N = N.copy()
        V = V.copy()
        
        A_indices = list(np.where(A_mask)[0])
        not_A_indices = [x for x in range(129) if not A_mask[x]]
        
        current_obj = get_objective(N)
        best_obj = current_obj
        best_A = A_mask.copy()
        best_N = N.copy()
        best_V = V.copy()
        
        no_improve = 0
        max_no_improve = 50
        
        while time.time() < time_limit:
            # Find best swap
            best_delta = None
            best_a = None
            best_b = None
            best_new_obj = current_obj
            
            # Sample a subset of swaps if sets are large to save time?
            # 129*129 = 16k swaps. Each eval is O(257). 4M ops.
            # In Python loop, this might be slow.
            # Let's vectorize the evaluation.
            
            # We want to find a in A, b not in A such that (peak, sum_sq) of N - V[a] + V[b] + delta_self[b] is minimized.
            
            # We can iterate over a in A.
            # For each a, we want best b.
            # This is still hard to vectorize fully because objective is max/sum_sq.
            
            # Let's just do a random sampling of swaps if full search is too slow.
            # Or, we can restrict to "promising" swaps.
            # Promising a: a is involved in high-count sums.
            # Promising b: b fills gaps.
            
            # For now, let's try full search but with a time check inside.
            
            improved = False
            # To speed up, we can limit the number of swaps evaluated per step
            # or use a heuristic.
            # Let's evaluate all swaps if < 1000, else sample.
            
            num_swaps = len(A_indices) * len(not_A_indices)
            
            if num_swaps <= 5000:
                candidates = [(a, b) for a in A_indices for b in not_A_indices]
            else:
                # Sample
                candidates = [(random.choice(A_indices), random.choice(not_A_indices)) for _ in range(5000)]
            
            random.shuffle(candidates)
            
            for a, b in candidates:
                # Compute delta N
                delta_N = -V[a] + V[b] + delta_self[b]
                # Note: V[a] includes self-pair of a. V[b] does not include self-pair of b.
                # So we add delta_self[b].
                # Do we need to subtract delta_self[a]?
                # V[a] already has the self-pair count (which is 1).
                # So -V[a] removes it.
                # So we don't need to subtract delta_self[a] explicitly.
                
                new_N = N + delta_N
                
                # Check validity and objective
                # We only care about target range for peak/missing
                new_target_N = new_N[target_min:target_max+1]
                
                # Check for negative counts (should not happen if logic is right)
                # if np.any(new_N < 0): continue
                
                missing = np.sum(new_target_N == 0)
                if missing > 0:
                    continue
                
                peak = int(np.max(new_target_N))
                sum_sq = int(np.sum(new_target_N**2))
                
                new_obj = (0, peak, sum_sq)
                
                # Compare
                if new_obj < best_new_obj:
                    best_new_obj = new_obj
                    best_a = a
                    best_b = b
                    best_delta = delta_N
                    improved = True
            
            if improved:
                # Apply move
                A_mask[best_a] = False
                A_mask[best_b] = True
                
                # Update V
                # V[x][x+a] -= 1 for all x
                # V[x][x+b] += 1 for all x
                for x in range(129):
                    V[x, x + best_a] -= 1
                    V[x, x + best_b] += 1
                
                N += best_delta
                
                A_indices.remove(best_a)
                if best_b not in not_A_indices:
                    not_A_indices.append(best_b)
                else:
                    not_A_indices.remove(best_b)
                A_indices.append(best_b)
                
                current_obj = best_new_obj
                no_improve = 0
                
                if current_obj < best_obj:
                    best_obj = current_obj
                    best_A = A_mask.copy()
                    best_N = N.copy()
                    best_V = V.copy()
            else:
                no_improve += 1
                if no_improve > max_no_improve:
                    break
        
        return best_A, best_N, best_V, best_obj

    def perturb(A_mask, V, N, k=5):
        """Randomly swap k elements."""
        A_mask = A_mask.copy()
        V = V.copy()
        N = N.copy()
        
        A_indices = list(np.where(A_mask)[0])
        not_A_indices = [x for x in range(129) if not A_mask[x]]
        
        for _ in range(k):
            if not A_indices or not not_A_indices:
                break
            a = random.choice(A_indices)
            b = random.choice(not_A_indices)
            
            # Apply swap
            A_mask[a] = False
            A_mask[b] = True
            
            for x in range(129):
                V[x, x + a] -= 1
                V[x, x + b] += 1
            
            delta_N = -V[a] + V[b] + delta_self[b] # Wait, V is updated?
            # No, V used in delta calculation should be the OLD V.
            # Let's calculate delta before updating V.
            
            # Redo:
            # This perturb function is getting messy. Let's just recalculate N and V after perturbation.
            # It's slower but safer.
            pass
            
        # Recalculate
        N = calculate_N(A_mask)
        V = calculate_V(A_mask)
        return A_mask, V, N

    best_peak = float('inf')
    best_A_mask = None
    best_N = None
    best_V = None
    best_sum_sq = float('inf')

    # Main Loop
    # Try different initial sizes
    for k in [30, 40, 50]:
        for i in range(5):
            if time.time() > deadline:
                break
                
            A_mask, N = greedy_init(k)
            V = calculate_V(A_mask)
            
            obj = get_objective(N)
            if obj[0] > 0:
                continue
                
            peak = obj[1]
            sum_sq = obj[2]
            
            if peak < best_peak or (peak == best_peak and sum_sq < best_sum_sq):
                best_peak = peak
                best_sum_sq = sum_sq
                best_A_mask = A_mask.copy()
                best_N = N.copy()
                best_V = V.copy()
                
            time_left = deadline - time.time()
            if time_left > 2:
                # Run local search
                A_opt, N_opt, V_opt, obj_opt = local_search(A_mask, N, V, time_left - 1)
                
                if obj_opt[0] == 0:
                    if obj_opt[1] < best_peak or (obj_opt[1] == best_peak and obj_opt[2] < best_sum_sq):
                        best_peak = obj_opt[1]
                        best_sum_sq = obj_opt[2]
                        best_A_mask = A_opt.copy()
                        best_N = N_opt.copy()
                        best_V = V_opt.copy()
                        
                # Perturb and search again if time allows
                if time.time() < deadline - 10:
                    A_pert, V_pert, N_pert = perturb(A_opt, V_opt, N_opt, k=5)
                    obj_pert = get_objective(N_pert)
                    if obj_pert[0] == 0:
                        A_opt2, N_opt2, V_opt2, obj_opt2 = local_search(A_pert, N_pert, V_pert, deadline - time.time() - 1)
                        if obj_opt2[0] == 0:
                            if obj_opt2[1] < best_peak or (obj_opt2[1] == best_peak and obj_opt2[2] < best_sum_sq):
                                best_peak = obj_opt2[1]
                                best_sum_sq = obj_opt2[2]
                                best_A_mask = A_opt2.copy()
                                best_N = N_opt2.copy()
                                best_V = V_opt2.copy()

    if best_A_mask is None:
        # Fallback
        A_mask = np.zeros(129, dtype=bool)
        for x in range(64, 129):
            A_mask[x] = True
        N = calculate_N(A_mask)
        obj = get_objective(N)
        if obj[0] == 0:
            best_peak = obj[1]
            best_sum_sq = obj[2]
            best_A_mask = A_mask.copy()
        else:
            return list(range(129))
        
    A_list = [i for i in range(129) if best_A_mask[i]]
    return A_list
