if U.shape[0] != n*m or V.shape[0] != m*p or W.shape[0] != p*n:
    raise ValueError("factor row counts must be n*m, m*p and p*n")
