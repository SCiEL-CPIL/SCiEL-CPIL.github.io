import numpy as np
import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 55  # safety margin for 60s limit
    
    n = 128
    MIN_SUM = 128
    MAX_SUM = 256
    
    class FastEvaluator:
        def __init__(self, A_set):
            self.A = set(A_set)
            self.conv = np.zeros(257, dtype=np.int32)
            for x in self.A:
                for y in self.A:
                    self.conv[x + y] += 1
            self._recompute_metrics()
        
        def _recompute_metrics(self):
            self.max_mult = 0
            self.covered = True
            for t in range(MIN_SUM, MAX_SUM + 1):
                ordered = self.conv[t]
                diag = 1 if (t % 2 == 0 and (t // 2) in self.A) else 0
                unordered = (ordered + diag) // 2
                if unordered == 0:
                    self.covered = False
                elif unordered > self.max_mult:
                    self.max_mult = unordered
        
        def add(self, x):
            if x in self.A:
                return False
            self.A.add(x)
            for y in self.A:
                self.conv[x + y] += 1
            self._recompute_metrics()
            return True
        
        def remove(self, x):
            if x not in self.A:
                return False
            self.A.remove(x)
            for y in self.A:
                self.conv[x + y] -= 1
            self._recompute_metrics()
            return True
        
        def get_score(self):
            return self.max_mult, self.covered
    
    # Better initial solution: sparse set
    A_init = set(range(0, 129, 3))
    A_init.add(128)
    
    current_eval = FastEvaluator(A_init)
    best_A = set(A_init)
    best_score, best_covered = current_eval.get_score()
    
    random.seed(42)
    
    T = 5.0
    T_min = 0.05
    cooling = 0.99995
    
    while time.time() < deadline:
        prev_score, prev_covered = current_eval.get_score()
        prev_A = set(current_eval.A)
        
        r = random.random()
        if r < 0.35 and len(current_eval.A) > 10:
            x = random.choice(tuple(current_eval.A))
            current_eval.remove(x)
        elif r < 0.7:
            x = random.randint(0, 128)
            if x not in current_eval.A:
                current_eval.add(x)
        else:
            if len(current_eval.A) > 5:
                x = random.choice(tuple(current_eval.A))
                y = random.randint(0, 128)
                if y not in current_eval.A:
                    current_eval.remove(x)
                    current_eval.add(y)
        
        new_score, new_covered = current_eval.get_score()
        
        if new_covered:
            new_cost = float(new_score)
        else:
            new_cost = float(new_score) + 10000.0
        
        if prev_covered:
            prev_cost = float(prev_score)
        else:
            prev_cost = float(prev_score) + 10000.0
        
        delta = new_cost - prev_cost
        
        if delta < 0 or (T > 0 and random.random() < np.exp(-delta / T)):
            if new_covered and new_score < best_score:
                best_score = new_score
                best_A = set(current_eval.A)
                best_covered = True
        else:
            for x in list(current_eval.A):
                if x not in prev_A:
                    current_eval.remove(x)
            for x in prev_A:
                if x not in current_eval.A:
                    current_eval.add(x)
        
        T *= cooling
        if T < T_min:
            T = T_min
    
    # Final local search polish
    if best_covered:
        final_eval = FastEvaluator(set(best_A))
        for _ in range(200):
            if time.time() > deadline:
                break
            improved = False
            for x in list(final_eval.A):
                if len(final_eval.A) <= 5:
                    break
                old_score, old_covered = final_eval.get_score()
                final_eval.remove(x)
                new_score, new_covered = final_eval.get_score()
                if new_covered and new_score <= old_score:
                    improved = True
                    if new_score < best_score:
                        best_score = new_score
                        best_A = set(final_eval.A)
                        best_covered = True
                else:
                    final_eval.add(x)
            if not improved:
                break
    
    if not best_covered:
        best_A = set(range(0, 129))
    
    return sorted(best_A)
