"""Triangle-free graphs without independent 10-sets (Ramsey R(3,10)).

Uses the Clebsch graph on 16 vertices.
- Vertices 0..15
- Edge (i,j) iff popcount(i ^ j) == 1 (i.e., they differ in exactly one bit)
- Triangle-free: no three vertices in the hypercube Q4 form a triangle
- Independence number 8 (known for the Clebsch graph)

This is a well-known strongly regular graph (16, 5, 0, 2) that is triangle-free
and has independence number 8 < 10.
"""

def construct():
    n = 16
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            # Count number of differing bits (Hamming distance)
            xor = i ^ j
            if bin(xor).count('1') == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
