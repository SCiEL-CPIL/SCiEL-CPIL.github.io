import numpy as np

def priority(adjacency):
    adj = adjacency.astype(float)
    degree = adj.sum(axis=1)
    # Sum of neighbor degrees
    neighbor_degree_sum = adj @ degree
    # Score: prefer low degree and low neighbor-degree-sum
    score = -(degree + neighbor_degree_sum)
    return score
