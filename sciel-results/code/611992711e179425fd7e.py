import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 85  # safety margin before 90s kill
    
    n = 128
    TARGET_START = 128
    TARGET_END = 256
    TARGETS = list(range(TARGET_START, TARGET_END + 1))
    
    class Solver:
        def __init__(self, A_init):
            self.v = np.zeros(129, dtype=np.int32)
            self.conv = np.zeros(257, dtype=np.int32)
            self.A = set(A_init)
            for x in self.A:
                self.v[x] = 1
            self.conv = np.convolve(self.v, self.v, mode='full')
        
        def add(self, x):
            if x in self.A:
                return False
            self.A.add(x)
            self.v[x] = 1
            # Update conv: C' = C + 2 * (v_shifted) + delta_2x
            # v_shifted[t] = v[t-x]
            # We need to update self.conv
            # self.conv[t] += 2 * self.v[t-x] for valid t-x
            # And self.conv[2x] += 1
            
            # Efficient update
            # Indices t such that t-x is in [0, 128] => t in [x, x+128]
            start_idx = x
            end_idx = min(256, x + 128)
            
            # The shift is x.
            # self.v[0..128-x] contributes to self.conv[x..128]
            # self.v[0..128] contributes to self.conv[x..x+128]
            
            # Let's just do it simply for correctness first, then optimize if needed.
            # Actually, numpy slicing is fast.
            # self.conv[x : x+129] += 2 * self.v
            # self.conv[2*x] += 1
            
            # Be careful with bounds
            end = x + 129
            if end > 257:
                end = 257
                v_slice = self.v[:257-x]
            else:
                v_slice = self.v
                
            self.conv[x:end] += 2 * v_slice
            if 2*x < 257:
                self.conv[2*x] += 1
                
            return True

        def remove(self, x):
            if x not in self.A:
                return False
            self.A.remove(x)
            self.v[x] = 0
            # Reverse of add
            end = x + 129
            if end > 257:
                end = 257
                v_slice = self.v[:257-x]
            else:
                v_slice = self.v
                
            self.conv[x:end] -= 2 * v_slice
            if 2*x < 257:
                self.conv[2*x] -= 1
            return True

        def swap(self, x_out, x_in):
            # Remove x_out, Add x_in
            # To avoid intermediate invalid states or double counting issues,
            # we can do it in one go if we are careful, but sequential is safer.
            self.remove(x_out)
            self.add(x_in)

        def get_score(self):
            """Return (max_multiplicity, covered)."""
            max_mult = 0
            covered = True
            # conv[t] is ordered count.
            # unordered count = (conv[t] + (1 if t/2 in A else 0)) / 2
            # We only care about t in [128, 256]
            
            # Vectorized check for coverage and max
            # t_indices = np.arange(128, 257)
            # diag_mask = np.zeros(257, dtype=bool)
            # diag_indices = 2 * np.array(list(self.A))
            # diag_indices = diag_indices[(diag_indices >= 128) & (diag_indices <= 256)]
            # diag_mask[diag_indices] = True
            
            # This loop is small (129 iterations), so Python loop is fine.
            for t in range(TARGET_START, TARGET_END + 1):
                ordered = self.conv[t]
                # Check if t/2 is in A
                if t % 2 == 0:
                    half = t // 2
                    diag = 1 if half in self.A else 0
                else:
                    diag = 0
                unordered = (ordered + diag) // 2
                if unordered == 0:
                    covered = False
                if unordered > max_mult:
                    max_mult = unordered
            return max_mult, covered

    def make_initials():
        initials = []
        # Dense sets
        initials.append(set(range(0, 129)))
        initials.append(set(range(64, 129)))
        # Structured
        for k in [2, 3, 4, 5, 6]:
            initials.append(set(range(0, 129, k)))
        # Random
        random.seed(42)
        for size in [40, 50, 60, 70, 80, 90, 100, 110]:
            initials.append(set(random.sample(range(0, 129), size)))
        return initials

    def local_search(A_init, time_budget, seed):
        random.seed(seed)
        solver = Solver(A_init)
        current_max, current_covered = solver.get_score()
        current_cost = current_max if current_covered else current_max + 10000
        
        best_max = current_max if current_covered else float('inf')
        best_covered = current_covered
        best_A = set(solver.A)
        
        T = 50.0
        T_min = 0.1
        cooling = 0.999
        
        while time.time() < time_budget and T > T_min:
            move_type = random.random()
            
            if len(solver.A) > 10 and move_type < 0.3:
                # Remove
                x = random.choice(list(solver.A))
                old_A = set(solver.A)
                solver.remove(x)
                new_max, new_covered = solver.get_score()
                new_cost = new_max if new_covered else new_max + 10000
                delta = new_cost - current_cost
                if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                    current_cost = new_cost
                    current_max = new_max
                    current_covered = new_covered
                    if current_covered and current_max < best_max:
                        best_max = current_max
                        best_A = set(solver.A)
                else:
                    # Revert
                    solver.add(x)
            elif move_type < 0.5:
                # Add
                x = random.randint(0, 128)
                if x in solver.A:
                    continue
                old_A = set(solver.A)
                solver.add(x)
                new_max, new_covered = solver.get_score()
                new_cost = new_max if new_covered else new_max + 10000
                delta = new_cost - current_cost
                if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                    current_cost = new_cost
                    current_max = new_max
                    current_covered = new_covered
                    if current_covered and current_max < best_max:
                        best_max = current_max
                        best_A = set(solver.A)
                else:
                    # Revert
                    solver.remove(x)
            else:
                # Swap
                if len(solver.A) == 0:
                    continue
                x_out = random.choice(list(solver.A))
                x_in = random.randint(0, 128)
                if x_in in solver.A and x_in != x_out:
                    continue
                
                old_A = set(solver.A)
                solver.swap(x_out, x_in)
                new_max, new_covered = solver.get_score()
                new_cost = new_max if new_covered else new_max + 10000
                delta = new_cost - current_cost
                if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
                    current_cost = new_cost
                    current_max = new_max
                    current_covered = new_covered
                    if current_covered and current_max < best_max:
                        best_max = current_max
                        best_A = set(solver.A)
                else:
                    # Revert
                    solver.swap(x_in, x_out) # Note: swap logic must be reversible or we restore from old_A
                    # Since swap modifies A, and we have old_A, let's just reset solver state if revert is complex
                    # Actually, swapping x_in and x_out restores the set if x_in was not in A and x_out was.
                    # If x_in was in A, we skipped. So x_in was NOT in A, x_out WAS.
                    # swap(x_out, x_in) removes x_out, adds x_in.
                    # swap(x_in, x_out) removes x_in, adds x_out. This restores.
                    
            T *= cooling
            
            # Occasionally restart if stuck? Or just continue.
            
        return best_A, best_max

    best_A = None
    best_score = float('inf')
    
    initials = make_initials()
    random.seed(42)
    random.shuffle(initials)
    
    # Allocate time to each initial
    # We have ~85 seconds.
    # Let's do a quick eval of initials first to prioritize.
    
    scored_initials = []
    for init in initials:
        if time.time() > start_time + 5:
            break
        s = Solver(init)
        sc, cov = s.get_score()
        cost = sc if cov else sc + 10000
        scored_initials.append((cost, init))
        
    scored_initials.sort(key=lambda x: x[0])
    
    # Run local search on top initials
    # Time budget per run:
    # If we have 10 initials, and 80s left, ~8s each.
    
    num_starts = len(scored_initials)
    if num_starts == 0:
        best_A = set(range(0, 129))
    else:
        for i, (cost, init) in enumerate(scored_initials):
            if time.time() >= deadline - 2:
                break
            remaining_starts = num_starts - i
            remaining_time = deadline - 2 - time.time()
            if remaining_time <= 0:
                break
            # Give more time to better initials?
            # Or equal time.
            time_alloc = max(1.0, remaining_time / remaining_starts)
            
            seed = 1000 + i
            ls_result, ls_score = local_search(init, time.time() + time_alloc, seed)
            
            if ls_score < best_score:
                best_score = ls_score
                best_A = ls_result
                
            # If we found a very good score, we might want to polish it further
            # But for now, just proceed to next.

    if best_A is None:
        best_A = set(range(0, 129))
        
    # Final verification
    s = Solver(best_A)
    sc, cov = s.get_score()
    if not cov:
        # Fallback to full set if somehow invalid
        best_A = set(range(0, 129))
        
    return sorted(best_A)
