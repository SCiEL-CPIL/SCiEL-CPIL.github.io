import random
import itertools

def construct():
    n = 20
    
    # Construct Paley(25) graph
    q = 25
    # Calculate quadratic residues mod 25
    residues = set()
    for i in range(1, q):
        residues.add((i * i) % q)
    
    # Build adjacency for 25 vertices
    adj_25 = [[0] * q for _ in range(q)]
    for u in range(q):
        for v in range(q):
            if u != v:
                diff = (v - u) % q
                if diff in residues:
                    adj_25[u][v] = 1
    
    # Select 20 vertices from 0..24
    # We pick the first 20 to be deterministic
    selected = list(range(20))
    
    # Map to 0..19
    adj = [[0] * n for _ in range(n)]
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            u = selected[i]
            v = selected[j]
            if adj_25[u][v] == 1:
                adj[i][j] = 1
                adj[j][i] = 1
                edges.append((i, j))
                
    # Convert to bitsets for fast common neighbor calculation
    # N[i] is an integer where bit j is 1 if edge (i, j) exists
    N = [0] * n
    for i in range(n):
        mask = 0
        for j in range(n):
            if adj[i][j]:
                mask |= (1 << j)
        N[i] = mask

    def popcount(x):
        return bin(x).count("1")

    def get_bad_triples():
        bad = []
        for u, v, w in itertools.combinations(range(n), 3):
            common = N[u] & N[v] & N[w]
            if popcount(common) > 2:
                bad.append((u, v, w, common))
        return bad

    # Local Search: Remove edges to fix violations
    # We iterate until no bad triples
    for _ in range(100): # Max iterations
        bad_triples = get_bad_triples()
        if not bad_triples:
            break
        
        # Score edges to remove
        # An edge (i, j) is a candidate if it helps break a common neighbor set
        # For a bad triple (u, v, w) with common neighbors C, 
        # removing (u, x) for x in C breaks x's common neighbor status.
        edge_scores = {}
        for u, v, w, common in bad_triples:
            # Iterate through set bits in common
            c = common
            while c:
                x = (c & -c).bit_length() - 1
                c &= c - 1
                
                # Edges (u, x), (v, x), (w, x) are candidates
                for y in (u, v, w):
                    if y != x and adj[y][x]:
                        e = (min(y, x), max(y, x))
                        edge_scores[e] = edge_scores.get(e, 0) + 1
        
        if not edge_scores:
            break
            
        max_score = max(edge_scores.values())
        candidates = [e for e, s in edge_scores.items() if s == max_score]
        u, v = random.choice(candidates)
        
        # Remove edge
        adj[u][v] = 0
        adj[v][u] = 0
        N[u] &= ~(1 << v)
        N[v] &= ~(1 << u)
        edges.remove((u, v)) if (u, v) in edges else edges.remove((v, u))

    # Convert to list of lists
    res = [[u, v] for u, v in edges]
    return res
