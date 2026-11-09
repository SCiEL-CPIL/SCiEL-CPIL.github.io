    def verify_decomposition(U, V, W, n, m, p):
        """Verify that the decomposition is exact."""
        rank = U.shape[1]
        for a in range(n):
            for b in range(m):
                for c in range(m):
                    for d in range(p):
                        for i in range(n):
                            for k in range(p):
                                lhs = 0.0
                                for t in range(rank):
                                    lhs += U[a*m+b][t] * V[c*p+d][t] * W[k*n+i][t]
                                rhs = 1.0 if (a == i and c == b and d == k) else 0.0
                                if abs(lhs - rhs) > 1e-9:
                                    return False
        return True
