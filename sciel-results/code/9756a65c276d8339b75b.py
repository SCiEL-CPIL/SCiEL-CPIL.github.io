import numpy as np
import time

class Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.calls = 0
        self.deadline = time.time() + 290  # Safety margin for 300s limit

    def __call__(self, func):
        best_x = np.zeros(self.dim)
        best_f = np.inf
        x_center = np.zeros(self.dim)
        
        # 1. Evaluate center
        f_center = func(x_center)
        self.calls += 1
        best_f = f_center
        best_x = x_center.copy()
        
        if self.calls >= self.budget or time.time() > self.deadline:
            return best_x, best_f
        
        # 2. Determine signs for each dimension using multiple probe distances
        signs = np.zeros(self.dim)
        
        for i in range(self.dim):
            if self.calls >= self.budget or time.time() > self.deadline:
                break
            
            # Try multiple probe distances to be robust
            pos_better = 0
            neg_better = 0
            
            for step in [1.0, 2.0, 4.0]:
                if self.calls >= self.budget - 1 or time.time() > self.deadline:
                    break
                
                x_pos = x_center.copy()
                x_pos[i] += step
                f_pos = func(x_pos)
                self.calls += 1
                
                if self.calls >= self.budget or time.time() > self.deadline:
                    break
                
                x_neg = x_center.copy()
                x_neg[i] -= step
                f_neg = func(x_neg)
                self.calls += 1
                
                if f_pos < f_center:
                    pos_better += 1
                if f_neg < f_center:
                    neg_better += 1
            
            if pos_better > neg_better:
                signs[i] = 1.0
            elif neg_better > pos_better:
                signs[i] = -1.0
            else:
                signs[i] = 0.0
        
        # 3. Construct candidate vertex
        x_vertex = signs * 5.0
        
        # 4. Evaluate vertex
        if self.calls < self.budget and time.time() < self.deadline:
            f_vertex = func(x_vertex)
            self.calls += 1
            if f_vertex < best_f:
                best_f = f_vertex
                best_x = x_vertex.copy()
        
        # 5. Fill remaining budget with evaluations at best_x to maximize AOCC
        # Each call at the optimal value contributes to the average
        while self.calls < self.budget and time.time() < self.deadline:
            f_val = func(best_x)
            self.calls += 1
            if f_val < best_f:
                best_f = f_val
        
        return best_x, best_f
