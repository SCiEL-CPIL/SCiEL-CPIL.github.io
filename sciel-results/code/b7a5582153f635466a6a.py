import itertools
import time

def construct():
    start_time = time.time()
    deadline = 110  # 110 seconds to be safe
    
    # 1. Construct Paley Graph of order 37
    p = 37
    n_paley = p
    # Squares mod 37
    squares = set()
    for x in range(1, p):
        squares.add((x * x) % p)
    
    # Adjacency matrix for Paley(37)
    # A[i][j] = 1 if j-i in squares
    A = [[0] * n_paley for _ in range(n_paley)]
    for i in range(n_paley):
        for j in range(i + 1, n_paley):
            if (j - i) % p in squares:
                A[i][j] = 1
                A[j][i] = 1
    
    # 2. Find all K4s in Paley(37)
    # A K4 is a set of 4 vertices {a,b,c,d} such that all 6 edges exist.
    # We can find triangles first, then check if the 4th vertex connects to all 3.
    # Or just brute force combinations of 4. C(37,4) = 66045. This is very small.
    k4s = []
    for quad in itertools.combinations(range(n_paley), 4):
        a, b, c, d = quad
        if A[a][b] and A[a][c] and A[a][d] and \
           A[b][c] and A[b][d] and \
           A[c][d]:
            k4s.append(quad)
    
    # 3. Find all independent sets of size 6 in Paley(37)
    # Independent set of size 6 means no edges between them.
    # C(37,6) = 2,324,784. This is larger but still manageable in Python if optimized?
    # 2.3 million iterations might take a few seconds.
    # We can prune: if a set of 5 is not independent, 6 containing it isn't.
    # But simple brute force might be okay.
    # Let's try to be smarter.
    # An independent set in G is a clique in G_complement.
    # G_complement is also Paley(37) (since -1 is a square mod 37? 37=1 mod 4, yes).
    # So we just need to find all cliques of size 6 in Paley(37).
    # Wait, Paley(37) has clique number 6?
    # If it has cliques of size 6, then the complement has independent sets of size 6.
    # We need to find all such sets.
    
    # Optimization: Find all cliques of size 6 in A.
    # We can use a recursive backtracking or just itertools with pruning.
    # Given the deadline, let's use itertools.combinations but break early if possible?
    # No, we need the list of all such sets to find the hitting set.
    
    # Actually, we don't need ALL sets. We just need to find a pair {u,v} that hits all of them.
    # We can iterate over pairs {u,v} and check if they hit all K4s and all alpha-6s.
    # Number of pairs: 666.
    # For each pair, check against all K4s (fast).
    # For each pair, check against all alpha-6s.
    # If we store the list of alpha-6s, it might be large.
    
    # Let's find the alpha-6s first.
    alpha6s = []
    # To speed up, we can find all independent sets of size 5, then extend to 6.
    # Or just use combinations.
    # Let's try combinations.
    for combo in itertools.combinations(range(n_paley), 6):
        is_independent = True
        for i in range(6):
            for j in range(i+1, 6):
                if A[combo[i]][combo[j]]:
                    is_independent = False
                    break
            if not is_independent:
                break
        if is_independent:
            alpha6s.append(combo)
            
    if time.time() > deadline:
        # Fallback to Paley(17)
        return _get_paley_17()

    # 4. Find a pair {u, v} that hits all K4s and all alpha6s
    # A pair {u, v} hits a set S if u in S or v in S.
    # We need {u, v} such that for all K in k4s, u in K or v in K.
    # And for all I in alpha6s, u in I or v in I.
    
    # We can iterate u from 0 to 36, v from u+1 to 36.
    found_pair = None
    for u in range(n_paley):
        for v in range(u + 1, n_paley):
            hits_all_k4 = True
            for k4 in k4s:
                if u not in k4 and v not in k4:
                    hits_all_k4 = False
                    break
            if not hits_all_k4:
                continue
            
            hits_all_alpha6 = True
            for a6 in alpha6s:
                if u not in a6 and v not in a6:
                    hits_all_alpha6 = False
                    break
            if not hits_all_alpha6:
                continue
            
            found_pair = (u, v)
            break
        if found_pair:
            break
            
    if found_pair:
        u, v = found_pair
        # Construct the graph on V \ {u, v}
        # Map old indices to new indices 0..34
        new_to_old = []
        old_to_new = {}
        for i in range(n_paley):
            if i != u and i != v:
                old_to_new[i] = len(new_to_old)
                new_to_old.append(i)
                
        edges = []
        for i in range(n_paley):
            if i == u or i == v:
                continue
            for j in range(i + 1, n_paley):
                if j == u or j == v:
                    continue
                if A[i][j]:
                    edges.append([old_to_new[i], old_to_new[j]])
                    
        return {'n': 35, 'edges': edges}
    else:
        # If no pair found, try Simulated Annealing for n=35?
        # Or just return Paley(17)
        return _get_paley_17()

def _get_paley_17():
    squares = {x * x % 17 for x in range(1, 17)}
    return {'n': 17, 'edges': [[i, j] for i in range(17) for j in range(i + 1, 17) if (j - i) % 17 in squares]}
