import time
import random

def construct():
    start_time = time.time()
    deadline = start_time + 290  # Safety margin
    
    # 1. Try Paley Graph on 41 vertices
    n = 41
    q = n
    # Check if q is prime and q % 4 == 1
    def is_prime(num):
        if num < 2: return False
        for i in range(2, int(num**0.5)+1):
            if num % i == 0: return False
        return True
        
    if is_prime(q) and q % 4 == 1:
        residues = {(x * x) % q for x in range(1, q)}
        # Build adjacency bitmasks
        adj = [0] * q
        for i in range(q):
            for j in range(i+1, q):
                if (j - i) % q in residues:
                    adj[i] |= (1 << j)
                    adj[j] |= (1 << i)
        
        if is_55_free(adj, q, start_time, deadline):
            edges = []
            for i in range(q):
                for j in range(i+1, q):
                    if (adj[i] >> j) & 1:
                        edges.append((i, j))
            return q, edges

    # 2. Try to construct a 42-vertex graph
    # Strategy: Start with Paley 41, add vertex 42.
    # Connect vertex 42 to a subset of 0..40.
    # We want to avoid creating K5s or independent sets of size 5.
    # This is complex. 
    # Alternative: Random search for 42 vertices.
    # Given time constraints, if Paley 41 fails, we might not have time for a robust 42-vertex search.
    # But let's try a quick local search on 42 vertices.
    
    n = 42
    # Initialize random graph
    adj = [0] * n
    for i in range(n):
        for j in range(i+1, n):
            if random.random() < 0.5:
                adj[i] |= (1 << j)
                adj[j] |= (1 << i)
                
    # Local search
    best_adj = adj
    best_violations = count_violations(adj, n)
    
    for step in range(1000):
        if time.time() > deadline:
            break
        # Find a violation
        viol = find_violation(adj, n)
        if viol is None:
            return n, adj_to_edges(adj, n)
        
        # Randomly flip an edge in the violation
        i, j = random.choice(viol)
        if (adj[i] >> j) & 1:
            adj[i] ^= (1 << j)
            adj[j] ^= (1 << i)
        else:
            adj[i] |= (1 << j)
            adj[j] |= (1 << i)
            
        # If this flip didn't help, revert? Or just keep going?
        # Simple hill climbing with random restarts is better.
        
    # Fallback to seed
    q = 37
    residues = {(x * x) % q for x in range(1, q)}
    return q, [(i, j) for i in range(q) for j in range(i + 1, q) if (j - i) % q in residues]

def is_55_free(adj, n, start_time, deadline):
    # Check K5
    # Use combinations of 5
    # This is slow in pure Python for n=41 if not optimized.
    # But 41 choose 5 is ~ 750k.
    # We can use a helper to check if 5 vertices form a clique.
    
    # Actually, let's just return False if we can't verify quickly.
    # But we can verify.
    
    from itertools import combinations
    
    # Check K5
    for combo in combinations(range(n), 5):
        if time.time() > deadline:
            return False # Timeout, assume invalid to be safe? Or assume valid? 
            # Safer to assume invalid if we can't prove valid.
        
        # Check if all pairs in combo are connected
        ok = True
        for i in range(5):
            for j in range(i+1, 5):
                u = combo[i]
                v = combo[j]
                if not (adj[u] >> v) & 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return False
            
    # Check Independent Set of Size 5 (K5 in complement)
    # Complement adj: ~adj[u] & mask
    mask = (1 << n) - 1
    for combo in combinations(range(n), 5):
        if time.time() > deadline:
            return False
            
        ok = True
        for i in range(5):
            for j in range(i+1, 5):
                u = combo[i]
                v = combo[j]
                if (adj[u] >> v) & 1:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            return False
            
    return True

def count_violations(adj, n):
    # Placeholder for local search
    return 0

def find_violation(adj, n):
    # Returns list of edges in a K5 or independent set
    from itertools import combinations
    for combo in combinations(range(n), 5):
        # Check K5
        is_clique = True
        is_indep = True
        for i in range(5):
            for j in range(i+1, 5):
                u = combo[i]
                v = combo[j]
                if (adj[u] >> v) & 1:
                    is_indep = False
                else:
                    is_clique = False
        if is_clique or is_indep:
            # Return edges of this combo
            edges = []
            for i in range(5):
                for j in range(i+1, 5):
                    edges.append((combo[i], combo[j]))
            return edges
    return None

def adj_to_edges(adj, n):
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            if (adj[i] >> j) & 1:
                edges.append((i, j))
    return edges
