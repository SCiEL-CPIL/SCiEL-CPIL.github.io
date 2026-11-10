"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    DEADLINE = start_time + 85.0  # 90s wall-clock, leave 5s margin
    
    N = 256
    max_sum = 2 * N
    perfect_squares = set()
    for k in range(1, int(math.isqrt(max_sum)) + 2):
        sq = k * k
        if sq <= max_sum:
            perfect_squares.add(sq)
    
    # Self-loop check: 2a is a perfect square
    self_loop_set = set()
    for m in range(1, 12):
        a = 2 * m * m
        if 1 <= a <= N:
            self_loop_set.add(a)
    
    remaining = [a for a in range(1, N + 1) if a not in self_loop_set]
    n_rem = len(remaining)
    vert_to_idx = {v: i for i, v in enumerate(remaining)}
    
    # Build adjacency as bitmasks
    adj_mask = [0] * n_rem
    for i in range(n_rem):
        a = remaining[i]
        mask = 0
        for b in remaining:
            if a != b and (a + b) in perfect_squares:
                j = vert_to_idx[b]
                mask |= (1 << j)
        adj_mask[i] = mask
        
    # Precompute degrees
    degrees = [bin(m).count('1') for m in adj_mask]
    
    def popcount(x):
        return bin(x).count('1')
    
    def greedy_min_degree(rs=None):
        if rs is None:
            rs = random.Random(0)
        available = (1 << n_rem) - 1
        result_indices = []
        
        while available:
            min_deg = n_rem + 1
            candidates = []
            avail_list = []
            temp = available
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                avail_list.append(i)
                temp ^= lsb
            
            for i in avail_list:
                deg = popcount(adj_mask[i] & available)
                if deg < min_deg:
                    min_deg = deg
                    candidates = [i]
                elif deg == min_deg:
                    candidates.append(i)
            
            v = rs.choice(candidates)
            result_indices.append(v)
            available &= ~(1 << v)
            available &= ~adj_mask[v]
        
        return result_indices

    def local_improvement(indices, rs, max_iters=500):
        """Fast local improvement using bitmasks with add and 2-swap moves"""
        if not indices:
            return indices
        
        selected_mask = 0
        for i in indices:
            selected_mask |= (1 << i)
        
        improved = True
        iters = 0
        
        while improved and iters < max_iters:
            improved = False
            iters += 1
            
            non_selected = ((1 << n_rem) - 1) & ~selected_mask
            non_sel_list = []
            temp = non_selected
            while temp:
                lsb = temp & (-temp)
                i = lsb.bit_length() - 1
                non_sel_list.append(i)
                temp ^= lsb
            
            rs.shuffle(non_sel_list)
            
            # Try adding vertices with no conflicts
            for v in non_sel_list:
                conflicts = adj_mask[v] & selected_mask
                if conflicts == 0:
                    selected_mask |= (1 << v)
                    improved = True
            
            # Try 2-swap: remove one selected, add one non-selected
            if not improved:
                sel_list = []
                temp2 = selected_mask
                while temp2:
                    lsb = temp2 & (-temp2)
                    i = lsb.bit_length() - 1
                    sel_list.append(i)
                    temp2 ^= lsb
                
                for v in non_sel_list:
                    conflicts = adj_mask[v] & selected_mask
                    if conflicts == 0:
                        continue
                    # Try removing each conflicting vertex and adding v
                    conf_list = []
                    tempc = conflicts
                    while tempc:
                        lsb = tempc & (-tempc)
                        i = lsb.bit_length() - 1
                        conf_list.append(i)
                        tempc ^= lsb
                    
                    for c in conf_list:
                        # Remove c, add v: check if v has other conflicts besides c
                        other_conf = adj_mask[v] & selected_mask & ~(1 << c)
                        if other_conf == 0:
                            # Also check c's other neighbors in selected are fine (they are, since we're removing c)
                            selected_mask = (selected_mask & ~(1 << c)) | (1 << v)
                            improved = True
                            break
                    if improved:
                        break
        
        result = []
        temp = selected_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            result.append(i)
            temp ^= lsb
        return result

    def simulated_annealing(start_mask, rs, max_iters=2000, T0=1.0, cooling=0.995):
        """Simulated Annealing to escape local optima"""
        current_mask = start_mask
        current_size = popcount(current_mask)
        best_mask = current_mask
        best_size = current_size
        T = T0
        
        # Precompute list of all vertices for random selection
        all_verts = list(range(n_rem))
        
        for _ in range(max_iters):
            if time.time() > DEADLINE:
                break
                
            # Pick a random vertex to flip (add or remove)
            v = rs.choice(all_verts)
            bit = 1 << v
            
            if current_mask & bit:
                # Try removing v
                new_mask = current_mask & ~bit
                new_size = current_size - 1
                delta = -1
            else:
                # Try adding v
                conflicts = adj_mask[v] & current_mask
                if conflicts != 0:
                    # "Remove C, Add v" move
                    conf_count = popcount(conflicts)
                    new_mask = (current_mask & ~conflicts) | bit
                    new_size = current_size - conf_count + 1
                    delta = new_size - current_size
                else:
                    new_mask = current_mask | bit
                    new_size = current_size + 1
                    delta = 1
            
            # Acceptance criterion
            if delta >= 0:
                current_mask = new_mask
                current_size = new_size
                if current_size > best_size:
                    best_size = current_size
                    best_mask = current_mask
            else:
                # Accept worse with probability exp(delta / T)
                if rs.random() < math.exp(delta / T):
                    current_mask = new_mask
                    current_size = new_size
            
            T *= cooling
            if T < 0.01:
                T = 0.01
                
        # Convert best_mask to list
        result = []
        temp = best_mask
        while temp:
            lsb = temp & (-temp)
            i = lsb.bit_length() - 1
            result.append(i)
            temp ^= lsb
        return result

    def iterated_sa(rs, max_rounds=50):
        """Run SA from multiple starts"""
        best_indices = []
        best_size = 0
        
        for r in range(max_rounds):
            if time.time() > DEADLINE:
                break
            
            # Start from greedy or random
            if r == 0:
                start = greedy_min_degree(rs)
            else:
                # Perturb best
                if best_indices:
                    k = rs.randint(2, 10)
                    to_remove = set(rs.sample(best_indices, min(k, len(best_indices))))
                    start = [x for x in best_indices if x not in to_remove]
                else:
                    start = greedy_min_degree(rs)
            
            start_mask = 0
            for i in start:
                start_mask |= (1 << i)
                
            # Local improve first
            improved_indices = local_improvement(start, rs, max_iters=200)
            improved_mask = 0
            for i in improved_indices:
                improved_mask |= (1 << i)
                
            # Then SA
            sa_indices = simulated_annealing(improved_mask, rs, max_iters=1000)
            
            if len(sa_indices) > best_size:
                best_size = len(sa_indices)
                best_indices = sa_indices
                
        return best_indices

    best_indices = []
    
    # Run multiple strategies
    for seed in range(50):
        if time.time() > DEADLINE:
            break
        rs = random.Random(seed + 100)
        indices = iterated_sa(rs, max_rounds=20)
        if len(indices) > len(best_indices):
            best_indices = indices

    # Final local improvement
    final_rs = random.Random(99999)
    best_indices = local_improvement(best_indices, final_rs, max_iters=1000)
    
    result = sorted([remaining[i] for i in best_indices])
    
    # Validate
    valid = True
    for i in range(len(result)):
        a = result[i]
        if (2 * a) in perfect_squares:
            valid = False
            break
        for j in range(i + 1, len(result)):
            b = result[j]
            if (a + b) in perfect_squares:
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        return [1]
    
    return result

# EVOLVE-BLOCK-END
