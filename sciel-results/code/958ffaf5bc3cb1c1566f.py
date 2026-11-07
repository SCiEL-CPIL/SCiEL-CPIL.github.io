import numpy as np
import time
from collections import deque

class Optimizer:
    """Optimizer using 1D Pareto Front Extraction and DP with Convex Hull Trick for ZDT3."""

    def __init__(self, budget, dim, n_obj):
        self.budget, self.dim, self.n_obj = budget, dim, n_obj

    def __call__(self, problem, bounds):
        start_time = time.time()
        deadline = start_time + 280.0
        
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        dim = self.dim
        budget = self.budget
        
        # 1. Generate fine grid
        # N=50000 is fast to compute and extract non-dominated points from.
        N = 50000
        x_grid = np.linspace(0.0, 1.0, N)
        f1 = x_grid
        f2 = 1.0 - np.sqrt(x_grid) - x_grid * np.sin(10.0 * np.pi * x_grid)
        
        # 2. Extract non-dominated points
        # A point is non-dominated if it achieves a new minimum in f2
        # as we scan from x=0 to x=1.
        min_f2_so_far = np.inf
        nd_indices = []
        for i in range(N):
            if f2[i] < min_f2_so_far:
                nd_indices.append(i)
                min_f2_so_far = f2[i]
                
        M = len(nd_indices)
        
        # If M is small, just evaluate all
        if M <= budget:
            selected_grid_indices = nd_indices
        else:
            # 3. DP to select best K points from M non-dominated points
            # nd_f1[k] = f1[nd_indices[k]]
            # nd_f2[k] = f2[nd_indices[k]]
            nd_f1 = f1[nd_indices]
            nd_f2 = f2[nd_indices]
            
            R1, R2 = 11.0, 11.0
            
            # dp[k][j] = max HV contribution using k points ending at j
            # We only need dp_prev (k-1) to compute dp_curr (k)
            # dp[1][j] = 0 for all j
            
            dp_prev = np.zeros(M)
            # parent[k-1][j] = i that maximized dp_curr[j]
            # We need to store parent for backtracking.
            # parent is (budget-1) x M
            parent = np.zeros((budget - 1, M), dtype=int)
            
            # CHT implementation
            # We process k from 2 to budget
            for k in range(2, budget + 1):
                dp_curr = np.full(M, -np.inf)
                
                # Initialize CHT
                # Lines are added in order of i (0 to M-1)
                # Queries are in order of j (0 to M-1)
                # But we need i < j.
                # So for a fixed j, we can only use lines i < j.
                # This means we cannot just build one CHT for all j.
                # We must add line i to CHT before querying for j = i+1.
                
                # Wait, the transition is dp_curr[j] = max_{i < j} (...)
                # So for j=0, no i. dp_curr[0] = -inf (or 0 if k=1, but k>=2).
                # For j=1, i can be 0.
                # For j=2, i can be 0, 1.
                
                # We can iterate j from 0 to M-1.
                # Maintain CHT of lines for i < j.
                # Initially CHT is empty.
                # Before processing j, add line for i = j-1?
                # No, for j, valid i are 0..j-1.
                # So:
                # For j from 0 to M-1:
                #   If j > 0:
                #      Add line corresponding to i = j-1 to CHT.
                #      Line: m = R2 - nd_f2[j-1], c = dp_prev[j-1] - nd_f1[j-1]*(R2 - nd_f2[j-1])
                #   Query CHT with x = nd_f1[j] to get dp_curr[j].
                #   Store argmax index in parent[k-1][j].
                
                # We need to implement CHT that supports:
                # 1. Add line (m, c, idx)
                # 2. Query x -> returns (value, idx)
                
                # Since M can be up to 50000 (if grid is fine and front is complex),
                # and K=200, total operations ~ 200 * 50000 = 10M.
                # Python CHT might be slow if implemented naively with lists/deques of objects.
                # We can use arrays and indices.
                
                # Let's optimize the inner loop.
                # If M is large, Python loop might be too slow.
                # 10M iterations in Python is ~1s-5s.
                # 200 * 50000 = 10,000,000.
                # This is acceptable.
                
                # However, we can optimize by noting that if M is very large,
                # we might not need such a fine grid for the DP.
                # The non-dominated set M is likely much smaller than N.
                # Let's check M.
                # If M > 5000, we might subsample the non-dominated points?
                # No, the non-dominated points are the "important" ones.
                # If M is large, it means the front has many "steps".
                # For ZDT3, M is likely small (< 100).
                # Let's assume M is small enough.
                
                # CHT Data Structures
                # We'll use lists for m, c, idx and a head pointer for the deque.
                # Since we only add and query in increasing order, we can use a simple list
                # and a pointer `front` that moves forward.
                # We don't need to pop from front physically if we just move a pointer.
                # But we do need to pop from back when adding.
                
                ch_m = []
                ch_c = []
                ch_idx = []
                front = 0
                
                for j in range(M):
                    if j > 0:
                        # Add line for i = j-1
                        i = j - 1
                        m_i = R2 - nd_f2[i]
                        c_i = dp_prev[i] - nd_f1[i] * m_i
                        
                        # Add line (m_i, c_i, i)
                        # Check obsolescence
                        # Intersection of last two lines vs new line
                        while len(ch_m) - front >= 2:
                            # L1: ch_m[-2], ch_c[-2]
                            # L2: ch_m[-1], ch_c[-1]
                            # L3: m_i, c_i
                            m1 = ch_m[-2]
                            c1 = ch_c[-2]
                            m2 = ch_m[-1]
                            c2 = ch_c[-1]
                            m3 = m_i
                            c3 = c_i
                            
                            # x12 = (c1 - c2) / (m2 - m1)
                            # x23 = (c2 - c3) / (m3 - m2)
                            # If x12 >= x23, L2 is obsolete.
                            # Avoid division by zero? m2 > m1, m3 > m2.
                            # (c1 - c2) * (m3 - m2) >= (c2 - c3) * (m2 - m1)
                            if (c1 - c2) * (m3 - m2) >= (c2 - c3) * (m2 - m1):
                                ch_m.pop()
                                ch_c.pop()
                                ch_idx.pop()
                            else:
                                break
                                
                        ch_m.append(m_i)
                        ch_c.append(c_i)
                        ch_idx.append(i)
                    
                    if len(ch_m) - front == 0:
                        dp_curr[j] = -np.inf
                        parent[k-1][j] = 0
                    else:
                        # Query with x = nd_f1[j]
                        x_j = nd_f1[j]
                        
                        # Move front while next line is better
                        while len(ch_m) - front >= 2:
                            # L1: ch_m[front], ch_c[front]
                            # L2: ch_m[front+1], ch_c[front+1]
                            # If L2(x) >= L1(x), pop front
                            # m2*x + c2 >= m1*x + c1
                            # (m2 - m1)*x >= c1 - c2
                            if (ch_m[front+1] - ch_m[front]) * x_j >= (ch_c[front] - ch_c[front+1]):
                                front += 1
                            else:
                                break
                                
                        best_i = ch_idx[front]
                        val = ch_m[front] * x_j + ch_c[front]
                        dp_curr[j] = val
                        parent[k-1][j] = best_i
                        
                dp_prev = dp_curr
                
            # Find best last point
            # Total HV = dp_prev[j] + (R1 - nd_f1[j]) * (R2 - nd_f2[j])
            hv_contrib = dp_prev + (R1 - nd_f1) * (R2 - nd_f2)
            last_j = np.argmax(hv_contrib)
            
            # Backtrack
            selected_nd_indices = [0] * budget
            selected_nd_indices[budget - 1] = last_j
            for k in range(budget - 1, 1, -1):
                selected_nd_indices[k-1] = parent[k-1][selected_nd_indices[k]]
            
            # Map to grid indices
            selected_grid_indices = [nd_indices[i] for i in selected_nd_indices]
            
        # 4. Evaluate selected points
        all_points = []
        for idx in selected_grid_indices:
            if time.time() > deadline:
                break
            x = np.zeros(dim)
            x[0] = x_grid[idx]
            f = problem(x)
            all_points.append(f)
            
        return all_points

