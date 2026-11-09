import numpy as np
import time
import math

def construct():
    t0 = time.time()
    N = 12
    M = 40
    total = 1 << N  # 4096

    masks = np.arange(total, dtype=np.int64)
    a_2d = masks[:, None]
    b_2d = masks[None, :]
    # is_sup[i, j] is 1 if mask i is a strict subset of mask j
    is_sup = ((a_2d & b_2d) == a_2d) & (a_2d != b_2d)
    is_sup_float = is_sup.astype(np.float64)

    # Precompute row sums or other properties? 
    # Score of a set S is sum_{i in S, j in S} is_sup[i, j]
    # This is x @ (is_sup @ x)

    def get_score(sel):
        # sel is a binary vector
        return float(sel @ (is_sup_float @ sel))

    def random_solution(rng):
        # Select M random distinct indices
        indices = rng.choice(total, size=M, replace=False)
        sel = np.zeros(total, dtype=np.float64)
        sel[indices] = 1.0
        return sel

    def swap_move(sel, rng):
        # Pick one selected and one unselected, swap them
        selected = np.where(sel > 0)[0]
        unselected = np.where(sel == 0)[0]
        
        if len(selected) == 0 or len(unselected) == 0:
            return None, None
            
        i = selected[rng.randint(len(selected))]
        j = unselected[rng.randint(len(unselected))]
        return int(i), int(j)

    def calculate_delta(sel, i, j):
        # We want to swap i (currently 1) with j (currently 0)
        # New sel: sel[i]=0, sel[j]=1
        # Score change:
        # Old contribution of i: sum_{k in S} is_sup[i,k] + sum_{k in S} is_sup[k,i]
        # Note: is_sup is not symmetric. 
        # Score = sum_{a,b} x_a is_sup[a,b] x_b
        # If we remove i and add j:
        # Delta = (New Score) - (Old Score)
        # Let's compute the interaction of i with the rest of the set (excluding i)
        # and interaction of j with the rest of the set (excluding j, but j is not in set)
        
        # Current set S contains i, does not contain j.
        # Let S' = S \ {i}
        # New set S_new = S' \cup {j}
        
        # Score(S) = Score(S') + sum_{k in S'} (is_sup[i,k] + is_sup[k,i]) + is_sup[i,i](0)
        # Score(S_new) = Score(S') + sum_{k in S'} (is_sup[j,k] + is_sup[k,j]) + is_sup[j,j](0)
        
        # Delta = sum_{k in S'} (is_sup[j,k] + is_sup[k,j]) - sum_{k in S'} (is_sup[i,k] + is_sup[k,i])
        
        # We can compute this using matrix-vector products.
        # Let v be the vector of current selection.
        # v_i = 1, v_j = 0.
        # S' corresponds to v with v_i=0.
        
        # sum_{k in S'} is_sup[i,k] = is_sup[i, :] @ (v - e_i) = is_sup[i, :] @ v - is_sup[i,i] = is_sup[i, :] @ v
        # sum_{k in S'} is_sup[k,i] = is_sup[:, i] @ (v - e_i) = is_sup[:, i] @ v - is_sup[i,i] = is_sup[:, i] @ v
        # So cost_i = is_sup[i, :] @ v + is_sup[:, i] @ v
        
        # Similarly for j (j is not in v, so v_j=0):
        # sum_{k in S'} is_sup[j,k] = is_sup[j, :] @ v
        # sum_{k in S'} is_sup[k,j] = is_sup[:, j] @ v
        # So gain_j = is_sup[j, :] @ v + is_sup[:, j] @ v
        
        # Delta = gain_j - cost_i
        
        v = sel
        cost_i = is_sup_float[i, :] @ v + is_sup_float[:, i] @ v
        gain_j = is_sup_float[j, :] @ v + is_sup_float[:, j] @ v
        return float(gain_j - cost_i)

    # Simulated Annealing
    rng = np.random.RandomState(42)
    
    # Initial solution: Greedy to get a good start
    sel = np.zeros(total, dtype=np.float64)
    # Simple greedy init
    for _ in range(M):
        if time.time() - t0 > 10: break
        vec = is_sup_float @ sel + is_sup_float.T @ sel
        vec[sel > 0] = -np.inf
        idx = int(np.argmax(vec))
        sel[idx] = 1.0
        
    current_score = get_score(sel)
    best_score = current_score
    best_sel = sel.copy()
    
    T_start = 100.0
    T_end = 0.1
    iterations = 50000 # Adjust based on time
    # We need to finish in 60s. 
    # Each iteration involves a few dot products of size 4096. 
    # 50000 * (2 dot products of 4096) is fast.
    
    # Actually, let's do a fixed number of iterations or time-based.
    max_time = 50.0
    step = 0
    while time.time() - t0 < max_time:
        T = T_start * (T_end / T_start) ** (step / iterations)
        
        i, j = swap_move(sel, rng)
        if i is None:
            break
            
        delta = calculate_delta(sel, i, j)
        
        # Accept if better or with probability
        if delta > 0:
            accept = True
        else:
            # Avoid overflow in exp
            if delta < -50 * T:
                accept = False
            else:
                accept = rng.rand() < math.exp(delta / T)
                
        if accept:
            sel[i] = 0.0
            sel[j] = 1.0
            current_score += delta
            if current_score > best_score:
                best_score = current_score
                best_sel = sel.copy()
        else:
            # If rejected, we still step, but maybe do a "restart" or just continue
            pass
            
        step += 1
        if step > iterations:
            # Reset temperature or stop? 
            # Let's just stop if we've done enough
            break

    # Convert best_sel to lists
    result = []
    for mask in np.where(best_sel > 0)[0]:
        result.append([b for b in range(N) if (int(mask) >> b) & 1])

    return result
