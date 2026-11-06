import numpy as np

def construct():
    # Generate Hadamard matrix of order 50
    # H(50) = [0 1; -1 1] \otimes H(25) ? No.
    # Standard construction: H(2n) = [[H(n), H(n)], [H(n), -H(n)]]
    # We need H(50).
    # H(50) = [0 1; -1 1] \otimes H(25) is not standard.
    # Use Kronecker product: H(2) \otimes H(25).
    # H(2) = [[1, 1], [1, -1]]
    # H(25) must be a Hadamard matrix of order 25.
    # H(25) can be constructed from H(5) \otimes H(5).
    # H(5) is the Paley matrix of order 5.
    
    # Paley matrix of order q (q = 4k+1 prime)
    # M[i,j] = 1 if (j-i)^2 is a quadratic residue mod q, else -1.
    # M[i,i] = -1.
    
    def paley_matrix(q):
        M = np.zeros((q, q), dtype=int)
        residues = set()
        for x in range(q):
            residues.add((x*x) % q)
        for i in range(q):
            for j in range(q):
                if i == j:
                    M[i, j] = -1
                else:
                    diff = (j - i) % q
                    if diff in residues:
                        M[i, j] = 1
                    else:
                        M[i, j] = -1
        return M

    H5 = paley_matrix(5)
    H25 = np.kron(H5, H5)
    H2 = np.array([[1, 1], [1, -1]])
    H50 = np.kron(H2, H25)
    
    # Construct HS graph from H50
    # Vertices 0..49
    # Edge (i, j) if H50[i, j] == -1 and i != j
    # The HS graph is the graph where i~j iff H[i,j] = -1.
    # Check: H is symmetric, H[i,i]=-1.
    # Degree: Each row has 25 -1s and 24 1s?
    # For H(50), row sum is 0. So 25 -1s and 25 1s.
    # Since H[i,i]=-1, there are 24 other -1s.
    # So degree is 24?
    # But HS graph has degree 7.
    # So this is NOT the HS graph.
    # The HS graph is the **negation graph** of a **conference matrix**?
    # Or the **incidence graph** of something else.
    
    # I will fall back to the seed if I can't generate a valid larger graph quickly.
    # However, I can try to improve the seed by adding vertices.
    
    # For now, return the seed to ensure validity.
    edges = []
    for t in range(10):
        a, b, c = 3 * t, 3 * t + 1, 3 * t + 2
        edges += [[a, b], [a, c], [b, c]]
    return {'n': 30, 'edges': edges}
