import math

MAX_MOD = 128

def _disjoint(r1, m1, r2, m2):
    g = math.gcd(m1, m2)
    return (r2 - r1) % g != 0

def construct():
    # Guaranteed valid small solution
    fallback = [[0, 2], [1, 4]]
    
    # Try to construct a larger solution using powers of 2
    # Moduli: 2, 4, 8, 16, 32, 64, 128
    moduli = [2, 4, 8, 16, 32, 64, 128]
    
    assignment = {}
    
    def dfs(idx):
        if idx == len(moduli):
            return True
        m = moduli[idx]
        for r in range(m):
            ok = True
            for i in range(idx):
                m_prev = moduli[i]
                r_prev = assignment[m_prev]
                if not _disjoint(r, m, r_prev, m_prev):
                    ok = False
                    break
            if ok:
                assignment[m] = r
                if dfs(idx + 1):
                    return True
                del assignment[m]
        return False

    if dfs(0):
        # Return in canonical format
        return [[assignment[m], m] for m in moduli]
    else:
        return fallback
