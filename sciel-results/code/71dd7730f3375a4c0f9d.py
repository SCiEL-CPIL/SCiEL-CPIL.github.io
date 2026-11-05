def construct():
            import numpy as np
            from scipy.optimize import milp, LinearConstraint, Bounds
            from fractions import Fraction
            
            denoms = list(range(2, 49))
            n = len(denoms)
            a = np.array([1.0 / d for d in denoms])
            
            # Objective: maximize sum a_i x_i -> minimize -sum a_i x_i
            c = -a
            
            # Constraint: sum a_i x_i <= 1 - 1e-12
            # A is shape (1, n)
            A = a.reshape(1, -1)
            constraint = LinearConstraint(A, -np.inf, 1.0 - 1e-12)
            
            integrality = np.ones(n)
            bounds = Bounds(0, 1)
            
            res = milp(c, constraints=constraint, integrality=integrality, bounds=bounds)
            
            if res.status == 0:
                x = res.x
                indices = [i for i in range(n) if x[i] > 0.5]
                
                # Verify
                s = Fraction(0)
                for i in indices:
                    s += Fraction(1, denoms[i])
                
                if s < 1:
                    return indices
                else:
                    # Should not happen with 1e-12 margin, but just in case
                    # Remove one item to make it strictly below?
                    # Find item with largest value?
                    # Or just return empty? No, must be non-empty.
                    # Try removing the smallest contribution?
                    # Actually, if s == 1, we can just pick a random subset that is known to be < 1.
                    # e.g., {1/2, 1/3, 1/7, 1/43} ...
                    pass
            
            # Fallback
            # Greedy
            s = Fraction(0)
            indices = []
            # Sort by value desc?
            # Just pick 1/2, 1/3, 1/7, 1/43?
            # 1/2 + 1/3 + 1/7 + 1/43 = (301 + 202 + 69 + 42) / 602 ... no
            # 1/2 + 1/3 + 1/8 + 1/12 + 1/24 ...
            # Let's just use a simple greedy in fallback
            vals = sorted(range(n), key=lambda i: -1.0/denoms[i])
            for i in vals:
                if s + Fraction(1, denoms[i]) < 1:
                    s += Fraction(1, denoms[i])
                    indices.append(i)
            if indices:
                return indices
            return [0] # 1/2 is always valid
