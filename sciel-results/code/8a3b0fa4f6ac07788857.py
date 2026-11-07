"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import itertools

def construct():
    n = 14
    m = 35
    values = list(range(1, n + 1))
    
    # Generate all possible edges
    all_edges = list(itertools.combinations(range(n), 2))
    
    # Calculate product and sum for each edge
    edge_info = []
    for i, j in all_edges:
        prod = values[i] * values[j]
        summ = values[i] + values[j]
        edge_info.append((prod, summ, i, j))
    
    # Sort by product to maximize collisions (small products collide more)
    edge_info.sort(key=lambda x: x[0])
    
    # Select the first 35 edges
    selected_edges = [e[2:] for e in edge_info[:m]]
    
    # Convert to list of lists as required
    edges = [list(e) for e in selected_edges]
    
    return {'values': values, 'edges': edges}
# EVOLVE-BLOCK-END
