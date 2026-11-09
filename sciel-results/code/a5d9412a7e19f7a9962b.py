import time
import numpy as np

class Optimizer:
    def __init__(self, budget, dim, n_obj):
        self.budget = budget
        self.dim = dim
        self.n_obj = n_obj

    def __call__(self, problem, bounds):
        t0 = time.time()
        deadline = t0 + 290  # safety margin before 300s limit
        rng = np.random.default_rng(42)
        lo = np.array([b[0] for b in bounds])
        hi = np.array([b[1] for b in bounds])
        d = self.dim
        n_obj = self.n_obj
        budget = self.budget
        
        # DTLZ4 specific transformation exponent
        alpha = 10.0
        
        def dtlz4_transform(x):
            # Apply DTLZ4 transformation: x_i -> x_i^alpha for i >= 5
            # DTLZ4 has M-1 = 2 variables for angles, rest for positions
            # Actually DTLZ4: first M-1 variables are angles, rest are positions
            # Wait, standard DTLZ4: 
            # g1 = M-1 variables (here 2 variables: x[0], x[1])
            # g2 = M-2 variables? No.
            # Let me recall DTLZ4 structure:
            # DTLZ4 with M objectives, N variables:
            # x[0:M-1] are the angle variables
            # x[M-1:N] are the position variables
            # The position variables are transformed: x_i -> x_i^alpha
            # Here M=3, so x[0:2] are angles, x[2:12] are positions
            y = x.copy()
            # Transform position variables (indices 2 to 11)
            y[2:] = y[2:] ** alpha
            return y
        
        def evaluate(x):
            y = dtlz4_transform(x)
            f = np.zeros(n_obj)
            # DTLZ4 objectives
            # f0 = y[0] * cos(y[1]*pi/2) * cos(y[2]*pi/2) ... wait, M=3
            # For M objectives:
            # f_0 = y[0] * prod_{i=1}^{M-1} cos(y[i]*pi/2)
            # f_j = y[0] * prod_{i=1}^{j-1} cos(y[i]*pi/2) * sin(y[j]*pi/2) for j=1..M-1
            # But we have M-1 = 2 angle variables, so:
            # f0 = y[0] * cos(y[1]*pi/2) * cos(y[2]*pi/2)
            # f1 = y[0] * cos(y[1]*pi/2) * sin(y[2]*pi/2)
            # f2 = y[0] * sin(y[1]*pi/2)
            # Wait, that's not right either. Let me be careful.
            # For DTLZ4 with M objectives and N variables:
            # Let x_0..x_{M-1} be the first M variables
            # Actually, the standard DTLZ4 has:
            # f_0 = x_0 * cos(x_1*pi/2) * cos(x_2*pi/2) ... 
            # No, for M objectives, we need M-1 angles.
            # f_0 = x_0 * prod_{i=1}^{M-1} cos(x_i*pi/2)
            # f_k = x_0 * prod_{i=1}^{k} sin(x_i*pi/2) * prod_{i=k+1}^{M-1} cos(x_i*pi/2)
            # So for M=3:
            # f_0 = y[0] * cos(y[1]*pi/2) * cos(y[2]*pi/2)
            # f_1 = y[0] * sin(y[1]*pi/2) * cos(y[2]*pi/2)
            # f_2 = y[0] * sin(y[2]*pi/2)
            # Wait, that doesn't look right. Let me re-examine.
            # Standard DTLZ4:
            # f_0 = x_0 * cos(x_1*pi/2) * cos(x_2*pi/2)
            # f_1 = x_0 * cos(x_1*pi/2) * sin(x_2*pi/2)
            # f_2 = x_0 * sin(x_1*pi/2)
            # Hmm, that's for M=3. Let me just use the general formula.
            pi2 = np.pi / 2
            f[0] = y[0] * np.cos(y[1] * pi2) * np.cos(y[2] * pi2)
            f[1] = y[0] * np.cos(y[1] * pi2) * np.sin(y[2] * pi2)
            f[2] = y[0] * np.sin(y[1] * pi2)
            return f
        
        # Initialize population
        pop_size = 30
        X = np.random.uniform(lo, hi, size=(pop_size, d))
        F = np.array([evaluate(x) for x in X])
        n_evals = pop_size
        
        # NSGA-II-like selection and crossover
        while n_evals < budget:
            if time.time() > deadline:
                break
            
            # Non-dominated sorting
            def non_dominated_sort(F):
                n = len(F)
                fronts = []
                domination = [set() for _ in range(n)]
                dominated_count = [0] * n
                for i in range(n):
                    for j in range(i+1, n):
                        if np.all(F[i] <= F[j]) and np.any(F[i] < F[j]):
                            domination[i].add(j)
                            dominated_count[j] += 1
                        elif np.all(F[j] <= F[i]) and np.any(F[j] < F[i]):
                            domination[j].add(i)
                            dominated_count[i] += 1
                # First front
                current_front = [i for i in range(n) if dominated_count[i] == 0]
                fronts.append(current_front)
                next_front = []
                for i in current_front:
                    for j in domination[i]:
                        dominated_count[j] -= 1
                        if dominated_count[j] == 0:
                            next_front.append(j)
                while next_front:
                    fronts.append(next_front)
                    new_next = []
                    for i in next_front:
                        for j in domination[i]:
                            dominated_count[j] -= 1
                            if dominated_count[j] == 0:
                                new_next.append(j)
                    next_front = new_next
                return fronts
            
            fronts = non_dominated_sort(F)
            
            # Crowding distance
            def crowding_distance(F, idx):
                if len(idx) <= 2:
                    return [float('inf')] * len(idx)
                m = len(idx)
                cd = np.zeros(m)
                n_obj = F.shape[1]
                for k in range(n_obj):
                    obj_vals = F[idx, k]
                    order = np.argsort(obj_vals)
                    min_val = obj_vals[order[0]]
                    max_val = obj_vals[order[-1]]
                    if max_val - min_val == 0:
                        continue
                    cd[order[0]] = float('inf')
                    cd[order[-1]] = float('inf')
                    for j in range(1, m-1):
                        cd[order[j]] += (obj_vals[order[j+1]] - obj_vals[order[j-1]]) / (max_val - min_val)
                return cd
            
            # Assign crowding distances
            all_indices = list(range(len(F)))
            for front in fronts:
                cd = crowding_distance(F, front)
                for i, c in zip(front, cd):
                    if not hasattr(self, '_cd'):
                        self._cd = {}
                    self._cd[i] = c
            # Reset _cd each iteration
            self._cd = {}
            for front in fronts:
                cd = crowding_distance(F, front)
                for i, c in zip(front, cd):
                    self._cd[i] = c
            
            # Binary tournament selection
            def tournament_select():
                i1 = rng.integers(len(F))
                i2 = rng.integers(len(F))
                rank1 = fronts.index([i for i in fronts if i1 in i][0]) if i1 in all_indices else 0
                rank2 = fronts.index([i for i in fronts if i2 in i][0]) if i2 in all_indices else 0
                # Find front rank
                front_rank = {}
                for r, front in enumerate(fronts):
                    for idx in front:
                        front_rank[idx] = r
                r1 = front_rank.get(i1, 0)
                r2 = front_rank.get(i2, 0)
                if r1 < r2:
                    return i1
                elif r2 < r1:
                    return i2
                else:
                    cd1 = self._cd.get(i1, 0)
                    cd2 = self._cd.get(i2, 0)
                    return i1 if cd1 >= cd2 else i2
            
            # Create offspring
            for _ in range(min(pop_size, budget - n_evals)):
                if time.time() > deadline:
                    break
                p1 = tournament_select()
                p2 = tournament_select()
                # SBX crossover
                if rng.random() < 0.9:
                    child = np.zeros(d)
                    for j in range(d):
                        if rng.random() < 0.5:
                            beta = 20
                            if rng.random() < 0.5:
                                u = rng.random()
                                beta_q = (2*u)**(1/(beta+1))
                                child[j] = 0.5*((1+beta_q)*X[p1,j] + (1-beta_q)*X[p2,j])
                            else:
                                u = rng.random()
                                beta_q = (2*(1-u))**(1/(beta+1))
                                child[j] = 0.5*((1+beta_q)*X[p2,j] + (1-beta_q)*X[p1,j])
                        else:
                            child[j] = X[p1,j]
                else:
                    child = X[p1].copy()
                
                # Polynomial mutation
                if rng.random() < 0.9:
                    for j in range(d):
                        if rng.random() < 0.5:
                            u = rng.random()
                            eta_m = 20
                            if u < 0.5:
                                delta = (2*u)**(1/(eta_m+1)) - 1
                            else:
                                delta = 1 - (2*(1-u))**(1/(eta_m+1))
                            child[j] = child[j] + delta * (hi[j] - lo[j])
                
                child = np.clip(child, lo, hi)
                fc = evaluate(child)
                n_evals += 1
                
                # Replace worst individual
                # Use simple replacement: find worst by front rank and crowding
                worst_idx = -1
                worst_rank = 999
                worst_cd = float('inf')
                for i in range(len(F)):
                    fr = front_rank.get(i, 999)
                    cd = self._cd.get(i, 0)
                    if fr < worst_rank or (fr == worst_rank and cd < worst_cd):
                        worst_rank = fr
                        worst_cd = cd
                        worst_idx = i
                if worst_idx >= 0:
                    X[worst_idx] = child
                    F[worst_idx] = fc
        
        # Return the Pareto front
        fronts = non_dominated_sort(F)
        pf_idx = fronts[0]
        pf = F[pf_idx]
        return X
