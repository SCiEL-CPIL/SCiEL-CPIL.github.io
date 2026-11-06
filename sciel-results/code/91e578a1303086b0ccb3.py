"""K4-free graph on 18 vertices minimizing max triangle-free induced subgraph."""
import random
import time
from itertools import combinations

N = 18
DEADLINE = time.time() + 140  # safety margin before 150s kill

def is_k4_free(adj):
    """Check if graph with adjacency matrix adj is K4-free."""
    for i in range(N):
        for j in range(i+1, N):
            if adj[i][j]:
                for k in range(j+1, N):
                    if adj[i][k] and adj[j][k]:
                        for l in range(k+1, N):
                            if adj[i][l] and adj[j][l] and adj[k][l]:
                                return False
    return True

def max_triangle_free_induced(adj):
    """Find the largest number of vertices inducing a triangle-free subgraph."""
    best = 0
    # For each subset of vertices, check if induced subgraph is triangle-free
    # Use incremental approach: build up from smaller subsets
    # For efficiency, check all subsets up to a certain size
    for size in range(N, best, -1):
        found = False
        for subset in combinations(range(N), size):
            # Check if induced subgraph on subset is triangle-free
            triangle_free = True
            for i in range(len(subset)):
                for j in range(i+1, len(subset)):
                    if adj[subset[i]][subset[j]]:
                        for k in range(j+1, len(subset)):
                            if adj[subset[i]][subset[k]] and adj[subset[j]][subset[k]]:
                                triangle_free = False
                                break
                        if not triangle_free:
                            break
                if not triangle_free:
                    break
            if triangle_free:
                found = True
                break
        if found:
            best = size
        else:
            break
    return best

def random_k4_free_graph():
    """Generate a random K4-free graph."""
    adj = [[False]*N for _ in range(N)]
    # Start with empty graph, add edges while maintaining K4-free
    for i in range(N):
        for j in range(i+1, N):
            if random.random() < 0.4:
                # Check if adding edge (i,j) creates a K4
                creates_k4 = False
                for k in range(N):
                    if k != i and k != j and adj[i][k] and adj[j][k]:
                        for l in range(N):
                            if l != i and l != j and l != k and adj[i][l] and adj[j][l] and adj[k][l]:
                                creates_k4 = True
                                break
                        if creates_k4:
                            break
                if not creates_k4:
                    adj[i][j] = True
                    adj[j][i] = True
    return adj

def local_search(adj, max_time=135):
    """Local search to minimize max triangle-free induced subgraph."""
    start = time.time()
    best_adj = [row[:] for row in adj]
    best_score = max_triangle_free_induced(adj)
    
    while time.time() < start + max_time:
        # Try flipping a random edge
        i = random.randint(0, N-1)
        j = random.randint(0, N-1)
        if i == j:
            continue
        if i > j:
            i, j = j, i
        
        # Try adding the edge
        if not adj[i][j]:
            # Check K4-free constraint
            creates_k4 = False
            for k in range(N):
                if k != i and k != j and adj[i][k] and adj[j][k]:
                    for l in range(N):
                        if l != i and l != j and l != k and adj[i][l] and adj[j][l] and adj[k][l]:
                            creates_k4 = True
                            break
                    if creates_k4:
                        break
            if not creates_k4:
                adj[i][j] = True
                adj[j][i] = True
                score = max_triangle_free_induced(adj)
                if score < best_score:
                    best_score = score
                    best_adj = [row[:] for row in adj]
                else:
                    # Revert
                    adj[i][j] = False
                    adj[j][i] = False
        else:
            # Try removing the edge
            adj[i][j] = False
            adj[j][i] = False
            score = max_triangle_free_induced(adj)
            if score < best_score:
                best_score = score
                best_adj = [row[:] for row in adj]
            else:
                # Revert
                adj[i][j] = True
                adj[j][i] = True
    
    return best_adj, best_score

def construct():
    """Construct a K4-free graph on 18 vertices minimizing max triangle-free induced subgraph."""
    global DEADLINE
    DEADLINE = time.time() + 140
    
    best_adj = None
    best_score = N  # worst case
    
    # Try multiple random starts
    while time.time() < DEADLINE:
        adj = random_k4_free_graph()
        if not is_k4_free(adj):
            continue
        
        score = max_triangle_free_induced(adj)
        if score < best_score:
            best_score = score
            best_adj = [row[:] for row in adj]
        
        # Do local search if we have time
        if time.time() < DEADLINE - 5:
            adj_ls, score_ls = local_search(adj, max_time=min(10, DEADLINE - time.time() - 2))
            if score_ls < best_score:
                best_score = score_ls
                best_adj = adj_ls
        
        if best_score <= 8:  # Good enough, stop early
            break
    
    if best_adj is None:
        # Fallback: complete tripartite K6,6,6
        parts = [list(range(0, 6)), list(range(6, 12)), list(range(12, 18))]
        edges = []
        for i in range(3):
            for j in range(i+1, 3):
                for a in parts[i]:
                    for b in parts[j]:
                        edges.append([a, b])
        return edges
    
    # Convert adjacency matrix to edge list
    edges = []
    for i in range(N):
        for j in range(i+1, N):
            if best_adj[i][j]:
                edges.append([i, j])
    return edges
