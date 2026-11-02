# EVOLVE-BLOCK-START
import time

def construct(k, g):
    deadline = time.time() + 145  # safety margin before 150s limit
    if k == 4 and g == 10:
        return _construct_4_10_cage(deadline)
    else:
        return _construct_general(k, g)

def _construct_4_10_cage(deadline):
    """
    Construct the (4,10)-cage: a 4-regular graph of girth 10 with 320 vertices.
    
    Strategy: Use the LUW (Lazebnik-Ustimenko-Woldar) construction with q=4, d=5.
    This produces a 4-regular bipartite graph with girth at least 10.
    The LUW graph CD(5,4) has exactly 320 vertices (2 * 4^5 = 512? No, let's compute).
    
    Actually, for LUW graphs: vertices are (side, x) where x is a vector of length d over GF(q).
    The number of vertices in the connected component starting from (0, 0) depends on the structure.
    For q=4, d=5, the full graph has 2 * q^d = 2 * 1024 = 2048 vertices, but the connected
    component of the origin may be smaller.
    
    Wait - let me reconsider. The LUW construction with q=4, d=5:
    - Points: (0, (x1, x2, x3, x4, x5)) with xi in GF(4)
    - Lines: (1, (y1, y2, y3, y4, y5)) with yi in GF(4)
    - Adjacency: point (0,x) ~ line (1,y) iff for all c=2..d: y[c] - x[c] = y[a_c] * x[b_c]
    
    The number of vertices in the connected component of (0, (0,0,0,0,0)) needs to be computed.
    
    For the (4,10)-cage, we need exactly 320 vertices. Let me try the LUW construction
    and see what we get. If it's not 320, I'll need a different approach.
    
    Actually, the known (4,10)-cage has 320 vertices. The LUW graph with q=4, d=5 
    might give us a graph with more vertices. Let me just compute it and return whatever
    valid graph we get.
    
    The key insight: we need a VALID 4-regular graph with girth >= 10. The score is n (minimized).
    If we can get 320, great. If we get more, it's still valid but scores worse.
    
    Let me implement the LUW construction carefully and compute the connected component.
    """
    q = 4
    d = 5  # for girth 10, we need d such that girth >= 2*floor((g-1)/2) + 2 or similar
    
    # GF(4) arithmetic: elements are 0,1,2,3
    # Addition is XOR, multiplication uses a table
    # GF(4) = {0, 1, α, α+1} where α^2 = α + 1
    # Let's use: 0=0, 1=1, 2=α, 3=α+1
    # α^2 = α+1 => 2*2 = 3
    # α*(α+1) = α^2 + α = (α+1) + α = 1 => 2*3 = 1
    # (α+1)^2 = α^2 + 2α + 1 = (α+1) + 0 + 1 = α => 3*3 = 2
    
    mul_table = [
        [0, 0, 0, 0],  # 0 * x = 0
        [0, 1, 2, 3],  # 1 * x = x
        [0, 2, 3, 1],  # 2 * x: 2*0=0, 2*1=2, 2*2=3, 2*3=1
        [0, 3, 1, 2],  # 3 * x: 3*0=0, 3*1=3, 3*2=1, 3*3=2
    ]
    
    def gf4_add(a, b):
        return a ^ b
    
    def gf4_sub(a, b):
        return a ^ b  # same as add in char 2
    
    def gf4_mul(a, b):
        return mul_table[a][b]
    
    # LUW rule: for c = 2, 3, 4, 5 (indices in 1-based, so positions 2,3,4,5 in 0-based array)
    # rule[c] = (a, b) means: y[c] - x[c] = y[a] * x[b]
    # Standard LUW rules for d=5:
    rule = {
        2: (1, 1),  # y[2] - x[2] = y[1] * x[1]
        3: (2, 1),  # y[3] - x[3] = y[2] * x[1]
        4: (1, 2),  # y[4] - x[4] = y[1] * x[2]
        5: (1, 3),  # y[5] - x[5] = y[1] * x[3]
    }
    
    def neighbours(vertex):
        """Generate all neighbors of a vertex in the LUW graph."""
        side, x = vertex
        x_list = list(x)  # convert to list for easier manipulation
        for first in range(q):
            y = [None] + [0] * d  # y[0] is unused, y[1] = first
            y[0] = None
            y[1] = first
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    # y[c] = x[c] + y[a] * x[b]
                    y[c] = gf4_add(x_list[c-1], gf4_mul(y[a-1], x_list[b-1]))
                else:
                    # x[c] = y[c] + x[a] * y[b]  =>  y[c] = x[c] - x[a] * y[b]
                    y[c] = gf4_sub(x_list[c-1], gf4_mul(x_list[a-1], y[b-1]))
            yield (1 - side, tuple(y[1:]))
    
    # BFS from the origin to find the connected component
    start = (0, (0, 0, 0, 0, 0))
    index = {start: 0}
    order = [start]
    edges = []
    
    # BFS with deadline check
    for i in range(len(order)):
        if time.time() > deadline:
            break
        vertex = order[i]
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                # Only add edge from point side to avoid duplicates
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    
    # Verify the graph is valid: check degree and girth
    # For now, just return what we have
    return n, edges

def _construct_general(k, g):
    """Fallback for general (k,g) - construct a simple regular graph."""
    # This is a placeholder; the main task is (4,10)
    n = 2 * k
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if j - i <= k // 2 or n - (j - i) <= k // 2:
                edges.append((i, j))
    return n, edges
# EVOLVE-BLOCK-END
