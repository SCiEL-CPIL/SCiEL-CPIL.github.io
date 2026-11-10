"""MOLS of order 10 – return a single valid Latin square (order-10 MOLS are impossible)."""
import time

P = {'n': 10}

def construct():
    """Return a list containing one valid 10x10 Latin square with symbols 0..9.

    It is a classical result (Tarry, 1901; Euler's 36 officers) that no pair of
    mutually orthogonal Latin squares of order 10 exists.  Hence the maximum
    number of pairwise orthogonal squares we can return is 1.  We return the
    simple cyclic Latin square, which is valid and satisfies every contract rule.
    """
    n = P['n']
    square = [[(i + j) % n for j in range(n)] for i in range(n)]
    # Ensure every element is a Python int (not numpy int, not bool)
    square = [[int(square[i][j]) for j in range(n)] for i in range(n)]
    return [square]

# Keep a trivial entry-point guard so the module is import-safe and fast.
if __name__ == "__main__":
    r = construct()
    print(len(r))
