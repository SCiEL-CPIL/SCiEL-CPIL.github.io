"""Triangle-free graphs without independent 10-sets (Ramsey R(3,10)).

Uses the Andrasfai graph And(9) on 26 vertices.
- Vertices 0..25
- Edge (i,j) iff (j - i) mod 26 is congruent to 1 mod 3
- Triangle-free: the circulant difference set {1,4,7,10,13,16,19,22,25} has no 3-term AP mod 26
- Independence number 9 (known for Andrasfai(9))
"""

def construct():
    n = 26
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if (j - i) % n % 3 == 1:
                edges.append([i, j])
    return {'n': n, 'edges': edges}
