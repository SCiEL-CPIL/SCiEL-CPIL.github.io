"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Assign a score to each vertex for a greedy maximum independent set decoder.
    
    Strategy: We want to pick vertices that are 'easy' to include, i.e., low degree,
    but we also want to avoid picking vertices that are neighbors of each other.
    A simple heuristic is 'degree', but better heuristics consider the structure.
    
    The greedy algorithm picks vertices in descending order of score.
    If we pick a vertex, we remove its neighbors.
    
    A common effective heuristic for MIS is to pick the vertex with the minimum degree
    first. Since the decoder picks in *descending* order of score, we should return
    negative degree (or a transformation that makes low degree = high score).
    
    However, simple degree is the seed. Let's try to improve it.
    
    One improvement: Consider the "weighted degree" or the impact on the remaining graph.
    If we pick vertex u, we lose all its neighbors. The "cost" of picking u is related to
    how many neighbors it has. But those neighbors might have high degrees themselves,
    meaning they would have been "expensive" to pick later or would have blocked many others.
    
    Actually, a standard result is that for random graphs, the greedy algorithm using
    minimum degree is quite good.
    
    Let's look at the "independent set size" objective.
    If we pick a vertex, we gain 1. We lose the opportunity to pick its neighbors.
    
    A more sophisticated score could be based on the "local" structure.
    For example, score(u) = -degree(u) is the seed.
    
    What if we use a score that accounts for the degrees of the neighbors?
    If a vertex has many high-degree neighbors, picking it removes those high-degree vertices,
    which might be a good thing because high-degree vertices are "bad" to keep (they block many).
    Wait, if I pick u, I remove N(u). The vertices in N(u) are gone.
    If the vertices in N(u) have high degrees, they were going to block many other vertices anyway.
    So removing them might be "efficient" in terms of clearing space?
    No, if I remove a high-degree vertex, I am "wasting" the potential of its neighbors?
    
    Let's think about the greedy process.
    1. Pick u with max score.
    2. Remove u and N(u).
    
    If u has low degree, it removes few vertices. This is generally good for maximizing the count.
    So low degree is good.
    
    Is there a reason to prefer a low-degree vertex over another low-degree vertex?
    Consider two vertices u and v, both degree 2.
    N(u) = {a, b}, N(v) = {c, d}.
    If deg(a) is high, then a is a "heavy" vertex.
    If I pick u, I remove a and b.
    If I pick v, I remove c and d.
    
    If a and b have high degrees, they are likely to be picked later if not removed?
    No, if they are not removed, they compete for selection.
    
    Actually, a known heuristic is "Minimum Sum of Neighbor Degrees" or similar.
    Let's try: score(u) = -degree(u) - alpha * sum(degrees of neighbors).
    If a vertex is connected to high-degree nodes, picking it removes those high-degree nodes.
    Removing high-degree nodes is "good" because they are "inefficient" to keep?
    Wait, if I keep a high-degree node, it blocks many. If I remove it, I free up its neighbors?
    No, if I remove it, I don't pick it. Its neighbors are still there (unless they are also neighbors of u).
    
    Let's re-evaluate.
    Objective: Maximize |IS|.
    Greedy: Pick u, remove N[u].
    
    If I pick u, I gain 1.
    The "damage" is that I cannot pick any v in N(u).
    Ideally, I want to pick u such that the vertices in N(u) are "worthless" or "hard to pick" anyway.
    Vertices with high degree are "hard to pick" in the sense that they conflict with many.
    But if I don't pick them, I might pick their neighbors?
    
    Actually, consider the expected number of vertices picked from a set S.
    It's roughly sum_{v in S} P(v is picked).
    
    A simpler approach: Use a score that is a function of degree, but maybe non-linear?
    Or use the "core number" or "k-core" properties?
    
    Let's try a score based on the degree of the vertex and the average degree of its neighbors.
    score(u) = -degree(u)
    This is the seed.
    
    Let's try: score(u) = -degree(u) * log(degree(u) + 1) ?
    Or score(u) = - (degree(u) + sum(deg(v) for v in N(u))) ?
    
    Let's test the hypothesis: "Picking a vertex removes its neighbors. If the neighbors have high degrees, they would have blocked many vertices if they were picked. By removing them, we 'clear' the graph efficiently?"
    Actually, if a neighbor has high degree, it is likely to be picked by the greedy algorithm if it has a high score (low degree? No, high degree = low score in seed).
    So high-degree vertices are picked *last*.
    If I pick a low-degree vertex u, and one of its neighbors v has high degree, v would have been picked late.
    By picking u, I remove v. This prevents v from being picked.
    Does this help or hurt?
    If v is removed, I don't get 1 for v.
    But v's neighbors are still there.
    If v had been picked, its neighbors would have been removed.
    So removing v early vs late might not change the set of removed vertices much, but the order matters for the greedy choice.
    
    There is a known heuristic: "Minimum Degree" is optimal for some cases, but "Maximum Degree" is bad.
    What about "Degree of Neighbors"?
    
    Let's look at the "Greedy Independent Set" literature.
    A common improvement is to use a score that considers the "local density".
    
    Let's try a score that penalizes vertices connected to high-degree vertices.
    score(u) = -degree(u) - 0.1 * sum(deg(v) for v in N(u))
    
    Why?
    If u is connected to high-degree vertices, those vertices are "important" blockers.
    If I pick u, I remove them.
    Wait, if I remove them, I lose the chance to pick them.
    But if I don't pick u, I might pick one of those high-degree vertices later?
    No, high-degree vertices have low priority (negative score).
    So they are picked late.
    If I pick u (low degree, high priority), I remove the high-degree neighbors.
    This prevents them from being picked.
    Since they have low priority, they would have been picked only if all high-priority vertices were gone.
    So removing them early is "safe" in the sense that I wasn't going to pick them soon anyway?
    Actually, if I don't remove them, they might survive until the end and be picked.
    So removing them *loses* a potential pick.
    Therefore, I should *avoid* picking vertices that are neighbors of high-degree vertices?
    Because picking u removes v (high degree). If I didn't pick u, v might still be picked later.
    So picking u is "expensive" because it kills a potential high-degree vertex.
    Therefore, I should prefer u with *low* sum of neighbor degrees.
    
    So: score(u) = -degree(u) - beta * sum(deg(v) for v in N(u))
    Here, high sum of neighbor degrees -> lower score (less likely to be picked).
    This means we prefer vertices whose neighbors are low-degree.
    This makes sense: we want to pick vertices that "sacrifice" low-value neighbors.
    
    Let's implement this.
    deg = adjacency.sum(axis=1)
    neighbor_deg_sum = adjacency @ deg  # sum of degrees of neighbors
    
    score = -deg - 0.5 * neighbor_deg_sum
    
    Let's add a time guard, though this is O(N^2) or O(N^3) depending on matrix multiplication.
    N=200. N^3 = 8,000,000. Very fast.
    
    We can also try to tune the beta parameter.
    
    Another idea: Use the "independent set" size estimation.
    But the score must be static (computed once per graph).
    
    Let's try the "sum of neighbor degrees" heuristic.
    
    Also, consider the "clique" structure? No, too complex.
    
    Let's try a combination:
    score = -deg - 0.1 * neighbor_deg_sum
    
    We can also normalize.
    
    Let's write the code.
"""

def priority(adjacency):
    # Convert to numpy array if not already
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency)
    
    # Ensure it's a boolean or int matrix for summing
    # The input is a bool matrix according to the contract.
    
    # Calculate degrees
    deg = adjacency.sum(axis=1).astype(float)
    
    # Calculate sum of degrees of neighbors
    # neighbor_deg_sum[i] = sum(deg[j] for j where adjacency[i,j] == True)
    # This is adjacency @ deg
    neighbor_deg_sum = adjacency.astype(float) @ deg
    
    # Heuristic:
    # We want to pick vertices with low degree.
    # We also want to pick vertices whose neighbors have low degree (so we don't "waste" high-degree neighbors).
    # score = -deg - alpha * neighbor_deg_sum
    
    # Let's try alpha = 0.5
    # If alpha is too high, we might ignore the degree of the vertex itself.
    
    # Let's experiment with alpha.
    # For ER graphs, degree distribution is Poisson.
    # Neighbor degrees are correlated with degree?
    
    # Let's try a simple linear combination.
    # score = -(deg + 0.5 * neighbor_deg_sum)
    
    # Alternative:
    # score = -deg * (1 + 0.1 * neighbor_deg_sum)
    
    # Let's stick to linear for stability.
    # We can also add a small random tie-breaker? No, must be deterministic for the same input?
    # The contract says "five generated Erdos-Renyi graphs". It doesn't specify if they are fixed or random each time.
    # "Mean independent-set size on five generated... graphs".
    # Usually, these are fixed test cases.
    # If they are random, the score should be robust.
    
    # Let's try:
    score = -deg - 0.5 * neighbor_deg_sum
    
    return score

# EVOLVE-BLOCK-END
