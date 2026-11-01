import time
import numpy as np
import random

def construct(q, n, radius):
    t0 = time.time()
    deadline = t0 + 110.0  # Safety margin for 120s limit (evaluator kills at 150s, but task says 120s)
    
    total = q ** n
    ball_size = 1 + n * (q - 1)
    
    # Precompute place values for index conversion
    base = [1] * n
    for i in range(n - 2, -1, -1):
        base[i] = base[i + 1] * q
        
    # Precompute all words as indices
    words = np.zeros((total, n), dtype=np.int32)
    for i in range(total):
        temp = i
        for j in range(n - 1, -1, -1):
            words[i, j] = temp % q
            temp //= q
            
    # Precompute ball_indices: for each word, list of indices within Hamming distance 1
    ball_indices = [[] for _ in range(total)]
    for i in range(total):
        w = words[i]
        ball_indices[i].append(i)
        for coord in range(n):
            orig = w[coord]
            for v in range(q):
                if v == orig:
                    continue
                new_idx = i + (v - orig) * base[coord]
                ball_indices[i].append(new_idx)
                
    # Convert to numpy arrays for faster vectorized operations
    ball_indices_np = [np.array(b, dtype=np.int32) for b in ball_indices]
    
    # Precompute ball membership matrix for fast swap validation
    # ball_mem[i, j] = True if j is in ball(i)
    ball_mem = np.zeros((total, total), dtype=bool)
    for i in range(total):
        ball_mem[i, ball_indices_np[i]] = True
    
    # Greedy construction
    def greedy_build():
        covered = np.zeros(total, dtype=bool)
        code = []
        
        while not np.all(covered):
            best_c = -1
            max_new = -1
            for c in range(total):
                ball = ball_indices_np[c]
                new_cov = np.sum(~covered[ball])
                if new_cov > max_new:
                    max_new = new_cov
                    best_c = c
                    if max_new == ball_size:
                        break
            code.append(best_c)
            covered[ball_indices_np[best_c]] = True
        return code
    
    # Initial greedy
    current_code = set(greedy_build())
    best_code = list(current_code)
    best_size = len(current_code)
    
    # Initialize counts
    cnt = np.zeros(total, dtype=np.int32)
    for c in current_code:
        cnt[ball_indices_np[c]] += 1
        
    def run_delete_phase(code_set, cnt_arr):
        """Removes redundant codewords. Returns True if any removed."""
        changed = False
        # Iterate over a copy to allow modification
        for c in list(code_set):
            ball = ball_indices_np[c]
            if np.min(cnt_arr[ball]) >= 2:
                code_set.remove(c)
                cnt_arr[ball] -= 1
                changed = True
        return changed
    
    def try_swap(c_remove, c_add, code_set, cnt_arr):
        """Try a swap. Returns True if successful."""
        if c_add in code_set:
            return False
        if c_remove == c_add:
            return False
            
        ball_rem = ball_indices_np[c_remove]
        ball_add = ball_indices_np[c_add]
        
        # Check if removing c_remove and adding c_add keeps coverage.
        # For x in ball_rem \ ball_add, we need cnt[x] >= 2.
        # Use precomputed ball_mem for fast membership check
        in_add = ball_mem[c_add]
        
        valid = True
        for x in ball_rem:
            if not in_add[x]:
                if cnt_arr[x] < 2:
                    valid = False
                    break
                    
        if not valid:
            return False
            
        # Perform swap
        code_set.remove(c_remove)
        code_set.add(c_add)
        cnt_arr[ball_rem] -= 1
        cnt_arr[ball_add] += 1
        return True

    # Simulated Annealing
    # We'll use a temperature schedule that allows accepting worse states to escape local minima
    # The "energy" is the number of codewords (we want to minimize)
    # But we also need to maintain coverage, so invalid states have infinite energy
    
    def simulated_annealing(code_set, cnt_arr, initial_temp=1.0, cooling_rate=0.995):
        """Simulated annealing to optimize code size."""
        current_energy = len(code_set)
        best_energy = current_energy
        best_code_copy = set(code_set)
        temp = initial_temp
        
        while time.time() < deadline - 10.0:
            # Run delete phase to remove redundant codewords
            while run_delete_phase(code_set, cnt_arr):
                current_energy = len(code_set)
                if current_energy < best_energy:
                    best_energy = current_energy
                    best_code_copy = set(code_set)
                
            # If we've hit a very good solution, break early
            if best_energy <= 160:
                break
            
            # Try a random move
            if len(code_set) == 0:
                break
                
            move_type = random.random()
            
            if move_type < 0.7:
                # Swap move: remove one, add one
                c_remove = random.choice(list(code_set))
                c_add = random.randint(0, total - 1)
                
                if try_swap(c_remove, c_add, code_set, cnt_arr):
                    new_energy = len(code_set)
                    delta_e = new_energy - current_energy
                    
                    # Accept if better, or accept worse with probability exp(-delta_e/temp)
                    if delta_e < 0 or random.random() < np.exp(-delta_e / max(temp, 1e-6)):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = current_energy
                            best_code_copy = set(code_set)
                    else:
                        # Reject: undo the swap
                        ball_rem = ball_indices_np[c_remove]
                        ball_add = ball_indices_np[c_add]
                        code_set.remove(c_add)
                        code_set.add(c_remove)
                        cnt_arr[ball_add] -= 1
                        cnt_arr[ball_rem] += 1
                        
            elif move_type < 0.85:
                # Insert move: add a codeword (may be rejected later by delete phase)
                c_add = random.randint(0, total - 1)
                if c_add not in code_set:
                    ball_add = ball_indices_np[c_add]
                    code_set.add(c_add)
                    cnt_arr[ball_add] += 1
                    
                    new_energy = len(code_set)
                    delta_e = new_energy - current_energy
                    
                    # Insertion always increases size, so only accept if we're in a "heating" phase
                    # or if we expect it to enable better swaps later
                    if random.random() < np.exp(-delta_e / max(temp, 1e-6)):
                        current_energy = new_energy
                    else:
                        # Reject: undo the insertion
                        code_set.remove(c_add)
                        cnt_arr[ball_add] -= 1
                        
            else:
                # Delete move: remove a codeword (only valid if coverage is maintained)
                c_remove = random.choice(list(code_set))
                ball_rem = ball_indices_np[c_remove]
                
                # Check if we can remove this codeword
                if np.min(cnt_arr[ball_rem]) >= 2:
                    code_set.remove(c_remove)
                    cnt_arr[ball_rem] -= 1
                    
                    new_energy = len(code_set)
                    delta_e = new_energy - current_energy
                    
                    if delta_e < 0 or random.random() < np.exp(-delta_e / max(temp, 1e-6)):
                        current_energy = new_energy
                        if current_energy < best_energy:
                            best_energy = new_energy
                            best_code_copy = set(code_set)
                    else:
                        # Reject: undo the deletion
                        code_set.add(c_remove)
                        cnt_arr[ball_rem] += 1
                        
            # Cool down
            temp *= cooling_rate
            
            # Periodically reheat to escape deep local minima
            if temp < 0.01 and random.random() < 0.05:
                temp = 0.5
            
            # Update best if we improved
            if best_energy <= 160:
                break
                
        return best_code_copy, best_energy
    
    # Run simulated annealing
    sa_code, sa_size = simulated_annealing(current_code, cnt)
    
    if sa_size < best_size:
        best_size = sa_size
        best_code = list(sa_code)
    
    # Final result conversion
    result = []
    for idx in best_code:
        w = []
        temp = idx
        for i in range(n):
            w.append(temp % q)
            temp //= q
        w.reverse()
        result.append(w)
        
    return result
