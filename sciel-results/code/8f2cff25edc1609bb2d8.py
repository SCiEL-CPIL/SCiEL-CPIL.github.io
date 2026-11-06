import numpy as np
from math import gcd

def construct():
    moduli = list(range(20, 201))
    n_mod = len(moduli)
    
    var_indices = {}
    total_vars = 0
    for m in moduli:
        for r in range(m):
            var_indices[(m, r)] = total_vars
            total_vars += 1
            
    c = np.zeros(total_vars)
    for m in moduli:
        val = -1.0 / m  # Minimize negative sum
        for r in range(m):
            c[var_indices[(m, r)]] = val
            
    try:
        from scipy.optimize import milp, LinearConstraint, Bounds
        from scipy.sparse import lil_matrix, csr_matrix, vstack
        
        A_rows = []
        b_list = []
        
        # Constraint 1: At most one residue per modulus
        for m in moduli:
            row = lil_matrix((1, total_vars))
            for r in range(m):
                row[0, var_indices[(m, r)]] = 1
            A_rows.append(row)
            b_list.append(1)
            
        # Constraint 2: Pairwise disjointness
        for i in range(n_mod):
            m_i = moduli[i]
            for j in range(i + 1, n_mod):
                m_j = moduli[j]
                g = gcd(m_i, m_j)
                for k in range(g):
                    row = lil_matrix((1, total_vars))
                    for r in range(m_i):
                        if r % g == k:
                            row[0, var_indices[(m_i, r)]] = 1
                    for r in range(m_j):
                        if r % g == k:
                            row[0, var_indices[(m_j, r)]] = 1
                    A_rows.append(row)
                    b_list.append(1)
                    
        A_ub = csr_matrix(vstack(A_rows))
        b_ub = np.array(b_list)
        
        bounds = Bounds(lb=0, ub=1)
        integrality = np.ones(total_vars)
        
        res = milp(
            c=c,
            constraints=LinearConstraint(A_ub, ub=b_ub),
            integrality=integrality,
            bounds=bounds
        )
        
        if res.status == 0:
            selected = []
            for m in moduli:
                for r in range(m):
                    if res.x[var_indices[(m, r)]] > 0.5:
                        selected.append([m, r])
                        break
            return selected
        else:
            # Fallback
            pass
            
    except Exception:
        pass
        
    # Fallback to Greedy
    selected = []
    for m in range(20, 201):
        found = False
        for r in range(m):
            valid = True
            for (m_i, r_i) in selected:
                g = gcd(m, m_i)
                if r % g == r_i % g:
                    valid = False
                    break
            if valid:
                selected.append((m, r))
                found = True
                break
    return [[m, r] for (m, r) in selected]
