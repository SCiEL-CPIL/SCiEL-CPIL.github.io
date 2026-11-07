"""BBOB f14 (Different powers) optimizer.

Best-First Multi-Start Nelder-Mead ensemble.
Prioritizes exploration of the most promising basin at each step to maximize
early global best improvements (AOCC).
"""
# EVOLVE-BLOCK-START
import numpy as np
import time


class Optimizer:
    """Best-First Multi-Start Nelder-Mead for BBOB f14 (Different powers)."""

    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        t0 = time.time()
        deadline = t0 + 290.0  # safety margin before 300s limit

        dim = self.dim
        budget = self.budget
        rng = np.random.default_rng(12345)

        calls = 0
        best_x = None
        best_f = np.inf

        def t_ok():
            return time.time() < deadline

        def ok():
            return t_ok() and calls < budget

        def record(x, f):
            nonlocal best_x, best_f
            if np.isfinite(f) and f < best_f:
                best_f = f
                best_x = x.copy()

        def call(x):
            nonlocal calls
            calls += 1
            return func(np.asarray(x, dtype=float))

        # Number of concurrent Nelder-Mead agents
        n_agents = 5
        
        # Latin Hypercube Sampling for diverse initial points
        lhs_points = []
        for d in range(dim):
            strata = np.linspace(-5, 5, n_agents + 1)
            for i in range(n_agents):
                lhs_points.append(rng.uniform(strata[i], strata[i+1]))
        
        lhs_points = np.array(lhs_points).reshape(n_agents, dim)
        
        # Initialize agents with LHS points and build initial simplexes
        agents = []
        for i in range(n_agents):
            x0 = lhs_points[i].copy()
            f0 = call(x0)
            record(x0, f0)
            
            # Build simplex
            simp = [x0.copy()]
            fs = [f0]
            
            step = 0.5
            for d in range(dim):
                if not ok():
                    # If we run out of budget during init, break out
                    # This is a rare edge case, but handle it
                    break
                xi = x0.copy()
                xi[d] += step
                xi = np.clip(xi, -5, 5)
                fi = call(xi)
                record(xi, fi)
                simp.append(xi.copy())
                fs.append(fi)
            
            if not ok():
                break
                
            agents.append({
                'simp': simp,
                'fs': fs,
                'f_best': min(fs),
                'active': True,
                'stagnation': 0
            })
        
        # If we couldn't even init all agents, adjust n_agents
        n_active_agents = len(agents)
        
        # Nelder-Mead parameters
        gamma = 1.0   # reflection
        rho = 2.0     # expansion
        sig = 0.5     # contraction
        shr = 0.5     # shrink
        
        # Best-First Loop
        while ok() and any(a['active'] for a in agents):
            # Find the active agent with the best current fitness
            best_agent_idx = -1
            best_agent_f = np.inf
            for i, a in enumerate(agents):
                if a['active'] and a['f_best'] < best_agent_f:
                    best_agent_f = a['f_best']
                    best_agent_idx = i
            
            if best_agent_idx == -1:
                break
                
            a = agents[best_agent_idx]
            simp = a['simp']
            fs = a['fs']
            
            # Sort simplex
            order = np.argsort(fs)
            simp = [simp[i] for i in order]
            fs = [fs[i] for i in order]
            
            x_best = simp[0]
            f_best = fs[0]
            x_worst = simp[-1]
            f_worst = fs[-1]
            f_second_worst = fs[-2]
            centroid = np.mean(simp[:-1], axis=0)
            
            # Reflection
            xr = centroid + gamma * (centroid - x_worst)
            xr = np.clip(xr, -5, 5)
            if not ok():
                break
            fr = call(xr)
            record(xr, fr)
            
            new_fs = None
            new_xr = None
            
            if fr < f_best:
                # Expansion
                xe = centroid + rho * (xr - centroid)
                xe = np.clip(xe, -5, 5)
                if not ok():
                    break
                fe = call(xe)
                record(xe, fe)
                if fe < fr:
                    simp[-1] = xe
                    fs[-1] = fe
                    a['f_best'] = min(fs)
                    a['stagnation'] = 0
                else:
                    simp[-1] = xr
                    fs[-1] = fr
                    a['f_best'] = min(fs)
                    a['stagnation'] = 0
            elif fr < f_second_worst:
                # Accept reflection
                simp[-1] = xr
                fs[-1] = fr
                a['f_best'] = min(fs)
                a['stagnation'] = 0
            else:
                # Contraction
                if fr < f_worst:
                    xc = centroid + sig * (xr - centroid)
                else:
                    xc = centroid - sig * (x_worst - centroid)
                xc = np.clip(xc, -5, 5)
                if not ok():
                    break
                fc = call(xc)
                record(xc, fc)
                if fc < min(fr, f_worst):
                    simp[-1] = xc
                    fs[-1] = fc
                    a['f_best'] = min(fs)
                    a['stagnation'] = 0
                else:
                    # Shrink
                    for i in range(1, len(simp)):
                        if not ok():
                            break
                        simp[i] = x_best + shr * (simp[i] - x_best)
                        simp[i] = np.clip(simp[i], -5, 5)
                        fs[i] = call(simp[i])
                        record(simp[i], fs[i])
                    a['f_best'] = min(fs)
                    a['stagnation'] += 1
                    
            # Update agent state
            a['simp'] = simp
            a['fs'] = fs
            
            # Convergence check
            if np.max(fs) - np.min(fs) < 1e-10:
                a['active'] = False
            elif a['stagnation'] > 20:
                # Restart this agent with a new random point if it's stuck
                # This helps escape local minima
                new_x0 = rng.uniform(-5, 5, dim)
                new_f0 = call(new_x0)
                record(new_x0, new_f0)
                
                new_sim = [new_x0.copy()]
                new_fs = [new_f0]
                
                step = 0.5
                for d in range(dim):
                    if not ok():
                        break
                    xi = new_x0.copy()
                    xi[d] += step
                    xi = np.clip(xi, -5, 5)
                    fi = call(xi)
                    record(xi, fi)
                    new_sim.append(xi.copy())
                    new_fs.append(fi)
                    
                if ok():
                    a['simp'] = new_sim
                    a['fs'] = new_fs
                    a['f_best'] = min(new_fs)
                    a['stagnation'] = 0
                    a['active'] = True
                else:
                    a['active'] = False

        return best_x, best_f
# EVOLVE-BLOCK-END
