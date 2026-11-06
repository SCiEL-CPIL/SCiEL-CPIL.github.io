"""K2,3-free graph on 20 vertices: maximize edges via structured init + bitmask greedy + local search + genetic algorithm."""
import time
import random

N = 20
DEADLINE_MARGIN = 15.0  # seconds before hard limit to stop


def _is_valid_addition(adj_bitmasks, u, v):
    """Check if adding edge (u,v) keeps graph K2,3-free."""
    for w in range(N):
        if w == u or w == v:
            continue
        if (adj_bitmasks[v] >> w) & 1:
            cnt = (adj_bitmasks[u] & adj_bitmasks[w]).bit_count()
            if cnt + 1 > 2:
                return False
        if (adj_bitmasks[u] >> w) & 1:
            cnt = (adj_bitmasks[v] & adj_bitmasks[w]).bit_count()
            if cnt + 1 > 2:
                return False
    return True


def _popcount(x):
    return x.bit_count()


def _count_edges(adj):
    total = 0
    for u in range(N):
        total += adj[u].bit_count()
    return total // 2


def _is_valid_graph(adj):
    """Check if graph is K2,3-free."""
    for u in range(N):
        for v in range(u + 1, N):
            if (adj[u] & adj[v]).bit_count() >= 3:
                return False
    return True


def _greedy_fill(adj, degrees, edge_set, deadline):
    """Greedily add edges to maximize density while maintaining K2,3-free."""
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    while time.time() < deadline:
        best_edge = None
        best_score = None
        
        for (u, v) in all_edges:
            if (u, v) in edge_set or (v, u) in edge_set:
                continue
            
            if not _is_valid_addition(adj, u, v):
                continue
            
            new_du = degrees[u] + 1
            new_dv = degrees[v] + 1
            max_deg = max(new_du, new_dv)
            score = (max_deg, degrees[u] + degrees[v])
            
            if best_score is None or score < best_score:
                best_score = score
                best_edge = (u, v)
        
        if best_edge is None:
            break
        
        u, v = best_edge
        adj[u] |= (1 << v)
        adj[v] |= (1 << u)
        degrees[u] += 1
        degrees[v] += 1
        edge_set.add((u, v))
        edge_set.add((v, u))
    
    return edge_set


def _local_search(adj, degrees, edge_set, deadline):
    """Try edge swaps: remove one edge, add up to two edges."""
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    current_count = _count_edges(adj)
    
    # Try single edge additions first
    changed = True
    while changed and time.time() < deadline:
        changed = False
        for (u, v) in all_edges:
            if (u, v) in edge_set:
                continue
            if _is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                edge_set.add((u, v))
                edge_set.add((v, u))
                current_count += 1
                changed = True
    
    # Try edge swaps: remove edge (a,b), add edges (c,d) and (e,f)
    while time.time() < deadline:
        improved = False
        for (a, b) in list(edge_set):
            if a > b:
                continue
            # Remove edge (a,b)
            adj[a] &= ~(1 << b)
            adj[b] &= ~(1 << a)
            degrees[a] -= 1
            degrees[b] -= 1
            edge_set.discard((a, b))
            edge_set.discard((b, a))
            
            # Try adding two edges
            added = []
            for (c, d) in all_edges:
                if (c, d) in edge_set or (d, c) in edge_set:
                    continue
                if c == a or c == b or d == a or d == b:
                    continue
                if _is_valid_addition(adj, c, d):
                    adj[c] |= (1 << d)
                    adj[d] |= (1 << c)
                    degrees[c] += 1
                    degrees[d] += 1
                    edge_set.add((c, d))
                    edge_set.add((d, c))
                    added.append((c, d))
                    if len(added) == 2:
                        break
            
            if len(added) == 2:
                # We removed 1, added 2, net +1
                improved = True
                current_count += 1
                # Keep this improvement
            else:
                # Revert: remove added edges
                for (c, d) in added:
                    adj[c] &= ~(1 << d)
                    adj[d] &= ~(1 << c)
                    degrees[c] -= 1
                    degrees[d] -= 1
                    edge_set.discard((c, d))
                    edge_set.discard((d, c))
                # Re-add original edge
                adj[a] |= (1 << b)
                adj[b] |= (1 << a)
                degrees[a] += 1
                degrees[b] += 1
                edge_set.add((a, b))
                edge_set.add((b, a))
        
        if not improved:
            break
    
    return current_count


def _structured_init():
    """Create a structured initial graph using 5-cliques with specific inter-clique connections."""
    adj = [0] * N
    degrees = [0] * N
    edge_set = set()
    
    # Create 5 cliques of size 4
    cliques = [[i*4 + j for j in range(4)] for i in range(5)]
    
    # Add intra-clique edges
    for clique in cliques:
        for i in range(len(clique)):
            for j in range(i+1, len(clique)):
                u, v = clique[i], clique[j]
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                edge_set.add((u, v))
                edge_set.add((v, u))
    
    # Add inter-clique edges: connect clique i to clique (i+1)%5 with a matching
    for i in range(5):
        clique_a = cliques[i]
        clique_b = cliques[(i+1) % 5]
        # Connect first 2 vertices of each clique
        for k in range(2):
            u = clique_a[k]
            v = clique_b[k]
            if _is_valid_addition(adj, u, v):
                adj[u] |= (1 << v)
                adj[v] |= (1 << u)
                degrees[u] += 1
                degrees[v] += 1
                edge_set.add((u, v))
                edge_set.add((v, u))
    
    # Also connect clique 0 to clique 2
    clique_a = cliques[0]
    clique_b = cliques[2]
    for k in range(2):
        u = clique_a[k+2]
        v = clique_b[k]
        if _is_valid_addition(adj, u, v):
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degrees[u] += 1
            degrees[v] += 1
            edge_set.add((u, v))
            edge_set.add((v, u))
    
    return adj, degrees, edge_set


def _genetic_algorithm(adj, degrees, edge_set, deadline):
    """
    Genetic algorithm with edge-bitset chromosomes.
    
    Each chromosome is a set of edges forming a K2,3-free graph.
    Crossover: select two parents, partition their edges into disjoint subsets,
    and combine them while maintaining K2,3-free validity.
    """
    pop_size = 12
    all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
    
    # Build initial population from current solution and random perturbations
    population = []
    
    # First individual: current solution
    population.append(set(edge_set))
    
    # Generate additional individuals via random restarts
    for _ in range(pop_size - 1):
        if time.time() >= deadline:
            break
        ind_edges = set()
        random.shuffle(all_edges)
        adj_temp = [0] * N
        for (u, v) in all_edges:
            if _is_valid_addition(adj_temp, u, v):
                adj_temp[u] |= (1 << v)
                adj_temp[v] |= (1 << u)
                ind_edges.add((u, v))
        if ind_edges:
            population.append(ind_edges)
    
    best_count = _count_edges(adj)
    best_edges = set(edge_set)
    
    # Genetic loop
    generation = 0
    while time.time() < deadline:
        generation += 1
        
        # Evaluate fitness
        def fitness(edges_set):
            # Count edges; all are valid by construction
            return len(edges_set)
        
        # Sort population by fitness (descending)
        population.sort(key=fitness, reverse=True)
        
        # Selection: tournament selection
        def select():
            candidates = random.sample(population, min(3, len(population)))
            return max(candidates, key=fitness)
        
        # Crossover: validity-preserving edge combination
        def crossover(parent1, parent2):
            """
            Combine edges from two parents while maintaining K2,3-free validity.
            Strategy: start with the better parent, try to add edges from the other
            parent that don't violate K2,3-free property.
            """
            # Start with the larger parent
            if len(parent1) >= len(parent2):
                base = set(parent1)
                donor = parent2
            else:
                base = set(parent2)
                donor = parent1
            
            # Build adjacency from base
            adj_temp = [0] * N
            for (u, v) in base:
                adj_temp[u] |= (1 << v)
                adj_temp[v] |= (1 << u)
            
            # Try to add edges from donor that aren't already in base
            child_edges = set(base)
            donor_edges = list(donor - base)
            random.shuffle(donor_edges)
            
            for (u, v) in donor_edges:
                if (u, v) in child_edges or (v, u) in child_edges:
                    continue
                if _is_valid_addition(adj_temp, u, v):
                    adj_temp[u] |= (1 << v)
                    adj_temp[v] |= (1 << u)
                    child_edges.add((u, v))
            
            return child_edges
        
        # Mutation: randomly add/remove edges
        def mutate(edges_set, rate=0.05):
            child = set(edges_set)
            adj_temp = [0] * N
            for (u, v) in child:
                adj_temp[u] |= (1 << v)
                adj_temp[v] |= (1 << u)
            
            # Try adding random edges
            for (u, v) in random.sample(all_edges, min(5, len(all_edges))):
                if (u, v) in child or (v, u) in child:
                    continue
                if random.random() < rate * 2:
                    if _is_valid_addition(adj_temp, u, v):
                        adj_temp[u] |= (1 << v)
                        adj_temp[v] |= (1 << u)
                        child.add((u, v))
            
            # Try removing random edges (less common)
            if child and random.random() < rate:
                to_remove = random.choice(list(child))
                u, v = to_remove
                adj_temp[u] &= ~(1 << v)
                adj_temp[v] &= ~(1 << u)
                child.discard(to_remove)
            
            return child
        
        # Generate new population
        new_population = []
        
        # Keep the best individual (elitism)
        new_population.append(population[0])
        
        for _ in range(pop_size - 1):
            if time.time() >= deadline:
                break
            
            p1 = select()
            p2 = select()
            
            # Crossover
            child = crossover(p1, p2)
            
            # Mutation
            if random.random() < 0.3:
                child = mutate(child)
            
            new_population.append(child)
        
        population = new_population
        
        # Track best
        for ind in population:
            cnt = fitness(ind)
            if cnt > best_count:
                best_count = cnt
                best_edges = set(ind)
        
        # Convergence check: if no improvement in several generations, break
        if generation > 50:
            break
    
    # Update adj, degrees, edge_set with best found
    if best_count > _count_edges(adj):
        for (u, v) in edge_set:
            adj[u] &= ~(1 << v)
            adj[v] &= ~(1 << u)
            degrees[u] -= 1
            degrees[v] -= 1
        edge_set.clear()
        
        for (u, v) in best_edges:
            adj[u] |= (1 << v)
            adj[v] |= (1 << u)
            degrees[u] += 1
            degrees[v] += 1
            edge_set.add((u, v))
            edge_set.add((v, u))
    
    return best_count


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE_MARGIN
    
    best_adj = [0] * N
    best_degrees = [0] * N
    best_edge_set = set()
    best_count = 0
    
    # Run multiple attempts with different strategies
    attempts = []
    
    # Attempt 1: Structured init + greedy fill
    adj1, deg1, es1 = _structured_init()
    es1 = _greedy_fill(adj1, deg1, es1, deadline)
    cnt1 = _count_edges(adj1)
    if cnt1 > best_count:
        best_adj = adj1[:]
        best_degrees = deg1[:]
        best_edge_set = es1.copy()
        best_count = cnt1
    
    # Attempt 2: Random restart with different edge orderings
    for attempt in range(3):
        if time.time() >= deadline:
            break
        
        adj2 = [0] * N
        deg2 = [0] * N
        es2 = set()
        
        # Randomized greedy
        all_edges = [(u, v) for u in range(N) for v in range(u + 1, N)]
        random.shuffle(all_edges)
        
        for (u, v) in all_edges:
            if time.time() >= deadline:
                break
            if (u, v) in es2:
                continue
            if _is_valid_addition(adj2, u, v):
                adj2[u] |= (1 << v)
                adj2[v] |= (1 << u)
                deg2[u] += 1
                deg2[v] += 1
                es2.add((u, v))
                es2.add((v, u))
        
        cnt2 = _count_edges(adj2)
        if cnt2 > best_count:
            best_adj = adj2[:]
            best_degrees = deg2[:]
            best_edge_set = es2.copy()
            best_count = cnt2
    
    # Local search on best solution
    if best_count > 0 and time.time() < deadline:
        local_deadline = min(deadline, time.time() + 20)
        best_count = _local_search(best_adj, best_degrees, best_edge_set, local_deadline)
    
    # Genetic algorithm on best solution
    if best_count > 0 and time.time() < deadline:
        ga_deadline = min(deadline, time.time() + 20)
        best_count = _genetic_algorithm(best_adj, best_degrees, best_edge_set, ga_deadline)
    
    # Final local search polish
    if best_count > 0 and time.time() < deadline:
        final_deadline = min(deadline, time.time() + 5)
        best_count = _local_search(best_adj, best_degrees, best_edge_set, final_deadline)
    
    # Convert to edge list
    edges = []
    for (u, v) in best_edge_set:
        if u < v:
            edges.append([u, v])
    
    # Final validity check
    if not _is_valid_graph(best_adj):
        # Fallback: return empty or minimal valid graph
        return []
    
    return edges
