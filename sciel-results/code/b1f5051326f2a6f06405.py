"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import time
import random
import itertools
import sys

# EVOLVE-BLOCK-START
def construct():
    """
    Try to find a (5,5)-free graph with as many vertices as possible.
    We start from the Paley graph on 37 vertices (known to be (5,5)-free) and
    attempt to add vertices one by one, or use a local search to improve.
    
    Since the exact value of R(5,5) is open (between 43 and 46), and the best
    known construction is Exoo's 42-vertex graph, we aim to find a graph with
    n > 37 vertices.
    
    Strategy:
    1. Start with the Paley graph on 37 vertices.
    2. Try to add new vertices by finding a valid connection pattern.
    3. Use a greedy/backtracking approach to extend the graph.
    
    However, extending a (5,5)-free graph is non-trivial. Instead, we can:
    - Try to construct a larger graph using known constructions or search.
    - Since we have a 300s budget, we can do a limited search.
    
    A simpler approach: use the Paley graph on 37 vertices as a seed and try
    to add vertices by checking all possible connection patterns.
    
    Actually, a more practical approach for this task:
    - The Paley graph on q vertices (where q is a prime congruent to 1 mod 4)
      is known to be (5,5)-free for small q. For q=37, it works.
    - We can try q=41, 43, 47, etc., but these may not be (5,5)-free.
    
    Let me check: the Paley graph on q vertices has no clique or independent
    set of size 5 if the maximum size of a quadratic residue class is small.
    
    Actually, a better approach: use a randomized search to find a (5,5)-free
    graph with more vertices.
    
    Given the time constraints, let's try a simpler approach:
    1. Start with the Paley graph on 37 vertices.
    2. Try to add one vertex at a time, checking if the resulting graph is still (5,5)-free.
    
    To check if a graph is (5,5)-free, we need to verify:
    - No clique of size 5
    - No independent set of size 5
    
    This is expensive for large graphs, but for n up to 42-46, it's feasible.
    
    Let's implement a function to check if a graph is (5,5)-free, and then
    use it to guide the search.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin before 300s limit
    
    # First, let's try to use the Paley graph on larger primes
    # and check if they are (5,5)-free
    
    def is_55_free(n, edges):
        """Check if graph with n vertices and given edges is (5,5)-free."""
        if n < 5:
            return True
        
        # Build adjacency matrix
        adj = [[False] * n for _ in range(n)]
        for i, j in edges:
            adj[i][j] = adj[j][i] = True
        
        # Check for clique of size 5
        for clique in itertools.combinations(range(n), 5):
            all_connected = True
            for a, b in itertools.combinations(clique, 2):
                if not adj[a][b]:
                    all_connected = False
                    break
            if all_connected:
                return False
        
        # Check for independent set of size 5
        for indep in itertools.combinations(range(n), 5):
            all_independent = True
            for a, b in itertools.combinations(indep, 2):
                if adj[a][b]:
                    all_independent = False
                    break
            if all_independent:
                return False
        
        return True
    
    # Try Paley graphs on primes q where q ≡ 1 (mod 4)
    # For q=37, it's known to work. Let's try larger ones.
    
    def paley_graph(q):
        """Generate Paley graph on q vertices."""
        if q < 2:
            return q, []
        residues = {(x * x) % q for x in range(1, q)}
        residues.discard(0)
        edges = []
        for i in range(q):
            for j in range(i + 1, q):
                if (j - i) % q in residues:
                    edges.append((i, j))
        return q, edges
    
    # Try various primes
    best_n = 0
    best_edges = []
    
    # First, ensure we have the 37-vertex Paley graph as fallback
    q37, edges37 = paley_graph(37)
    if is_55_free(q37, edges37):
        best_n = q37
        best_edges = edges37
    
    # Try to find larger (5,5)-free graphs
    # Try Paley graphs on other primes
    for q in [41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]:
        if time.time() > deadline:
            break
        
        # q must be a prime congruent to 1 mod 4 for Paley graph to be defined
        if q % 4 != 1:
            continue
        
        n, edges = paley_graph(q)
        # Only check if it might be (5,5)-free (heuristic: check a subset first)
        # For efficiency, let's just check the full graph
        if is_55_free(n, edges):
            if n > best_n:
                best_n = n
                best_edges = edges
    
    # If we haven't found anything better, try a randomized approach
    # to extend the 37-vertex graph
    
    if best_n <= 37 and time.time() < deadline - 30:
        # Try to add vertices to the Paley graph on 37
        n, edges = paley_graph(37)
        adj = [[False] * n for _ in range(n)]
        for i, j in edges:
            adj[i][j] = adj[j][i] = True
        
        # Try to add vertices one by one
        for new_v in range(38, 47):
            if time.time() > deadline - 10:
                break
            
            # Try all possible connection patterns for the new vertex
            # This is 2^37 possibilities, which is too many.
            # Instead, use a heuristic: try to connect to a random subset
            
            found = False
            for _ in range(1000):  # Try 1000 random patterns
                if time.time() > deadline - 5:
                    break
                
                # Randomly decide connections
                new_adj = [random.random() < 0.5 for _ in range(n)]
                
                # Check if adding this vertex creates a K5 or independent 5-set
                valid = True
                
                # Check cliques involving the new vertex
                for combo in itertools.combinations(range(n), 4):
                    if all(new_adj[i] for i in combo):
                        # Check if the 4 vertices form a K4
                        k4 = True
                        for a, b in itertools.combinations(combo, 2):
                            if not adj[a][b]:
                                k4 = False
                                break
                        if k4:
                            valid = False
                            break
                
                if not valid:
                    continue
                
                # Check independent sets involving the new vertex
                for combo in itertools.combinations(range(n), 4):
                    if all(not new_adj[i] for i in combo):
                        # Check if the 4 vertices form an independent set
                        indep4 = True
                        for a, b in itertools.combinations(combo, 2):
                            if adj[a][b]:
                                indep4 = False
                                break
                        if indep4:
                            valid = False
                            break
                
                if valid:
                    # Add the vertex
                    for i in range(n):
                        if new_adj[i]:
                            edges.append((i, new_v))
                            adj[i][new_v] = adj[new_v][i] = True
                    n = new_v + 1
                    found = True
                    break
            
            if not found:
                break
        
        if n > best_n:
            best_n = n
    
    return best_n, best_edges
# EVOLVE-BLOCK-END
