import numpy as np

def priority(adjacency):
    """
    Assigns a score to each vertex to guide a greedy Maximum Independent Set algorithm.
    
    The heuristic prioritizes vertices with a low 'closed neighborhood degree sum'.
    A vertex u with a low sum of degrees in its closed neighborhood N[u] is
    considered 'peripheral' and safe to pick, as it blocks fewer high-degree vertices.
    
    Score(u) = - ( degree(u) + sum_{v in neighbors(u)} degree(v) )
    """
    adj = np.asarray(adjacency)
    
    # Compute degrees for all vertices
    degrees = adj.sum(axis=1).astype(float)
    
    # Compute the sum of degrees of neighbors for each vertex
    # This is a matrix-vector product: adj (n x n) @ degrees (n,) -> (n,)
    neighbor_deg_sum = adj.dot(degrees)
    
    # Calculate the total weight of the closed neighborhood
    # W(u) = d(u) + sum_{v in N(u)} d(v)
    weights = degrees + neighbor_deg_sum
    
    # Return negative weights so that greedy selection in descending order
    # picks vertices with the lowest weights first.
    return -weights
