<<<<<<< SEARCH
    def reduce_pairs(U, V, W, max_rounds=20):
        """Try dropping pairs of columns."""
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            r = U.shape[1]
            dropped = False
            for i in range(r):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, r):
                    if time.time() > deadline - 10:
                        break
                    res = try_drop_columns(U, V, W, [i, j])
                    if res is not None:
                        U, V, W = res
                        dropped = True
                        break
                if time.time() > deadline - 10:
                    break
                if dropped:
                    break
            if not dropped:
                break
        return U, V, W
=======
    def reduce_pairs(U, V, W, max_rounds=20):
        """Try dropping pairs of columns."""
        for _ in range(max_rounds):
            if time.time() > deadline - 10:
                break
            r = U.shape[1]
            dropped = False
            for i in range(r):
                if time.time() > deadline - 10:
                    break
                for j in range(i + 1, r):
                    if time.time() > deadline - 10:
                        break
                    res = try_drop_columns(U, V, W, [i, j])
                    if res is not None:
                        U, V, W = res
                        dropped = True
                        break
                if time.time() > deadline - 10:
                    break
                if dropped:
                    break
            if not dropped:
                break
        return U, V, W

    def reduce_linear(U, V, W, max_rounds=50):
        """
        Try to find a column that is a linear combination of others in a
        tensor-compatible way. For each column c, check if there exist coefficients
        alpha_r such that for all (i,j,k):
        U[i,j,c]*V[j,k,c]*W[k,i,c] = sum_r alpha_r * U[i,j,r]*V[j,k,r]*W[k,i,r]
        
        This is a hard nonlinear problem, so we use a heuristic:
        For each column c, try to express the "pattern" of column c as a
        combination of other columns' patterns.
        """
        for _ in range(max_rounds):
            if time.time() > deadline - 15:
                break
            r = U.shape[1]
            if r <= 1:
                break
            dropped = False
            for c in range(r):
                if time.time() > deadline - 15:
                    break
                # Try to drop column c by checking if the remaining columns
                # still satisfy the tensor equation
                keep = [x for x in range(r) if x != c]
                U2, V2, W2 = U[:, keep].copy(), V[:, keep].copy(), W[:, keep].copy()
                if verify(U2, V2, W2):
                    U, V, W = U2, V2, W2
                    dropped = True
                    break
            if not dropped:
                break
        return U, V, W
>>>>>>> REPLACE

