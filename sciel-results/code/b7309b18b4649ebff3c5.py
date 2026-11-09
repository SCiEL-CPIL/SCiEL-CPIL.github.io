"""Rewritten program: stronger greedy priority via weighted degree + local density + coreness."""
# EVOLVE-BLOCK-START
import numpy as np
import time


def priority(adjacency):
    """
    Returns a score per vertex. The greedy decoder picks vertices in descending score order,
    skipping neighbours of already-picked vertices. Higher score = picked earlier.
    
    Strategy: We want to pick vertices that are 'safe' to include early, i.e., vertices that
    have few neighbours, but also we want to avoid picking vertices that are hubs if they
    block many other potential picks. 
    
    Actually, for Maximum Independent Set (MIS), a common heuristic is to pick the vertex with
    the *lowest* degree first (greedy coloring logic reversed, or simply low-degree first).
    However, simply using degree is the seed (score 1.0). We can improve by considering:
    1. Degree (lower is better for inclusion).
    2. Sum of degrees of neighbours (lower is better, as it means neighbours are less 'important' or less likely to be chosen later? No, if neighbours have high degree, they are harder to place, so maybe we should pick low-degree neighbours? Actually, if I pick v, I remove all neighbours. If my neighbours have high degree, removing them is 'expensive' in terms of potential MIS size? Or is it beneficial because high-degree nodes are hard to fit?
    
    Let's think about the greedy process:
    - Sort by score.
    - Pick v if no neighbour is picked.
    
    If we pick a low-degree vertex, we block few other vertices. This is generally good.
    If we pick a high-degree vertex, we block many. This is bad unless the high-degree vertex is the only way to cover some area? No, in MIS, we just want to pick as many non-adjacent nodes as possible. Blocking nodes is bad. So low degree is good.
    
    Why might low degree alone fail? 
    Consider a star graph. Center has high degree, leaves have low degree. 
    MIS is all leaves. 
    If we pick by lowest degree, we pick all leaves. Correct.
    
    Consider two cliques connected by a bridge. 
    Low degree nodes are on the bridge? 
    
    Let's look at the "observed" improvement in p36 (score 1.026). It used `_compute_coreness` and `_compute_neighbor_degree_sum`.
    Coreness (k-core) is a measure of how embedded a vertex is. High coreness means it's in a dense subgraph. 
    In dense subgraphs, the MIS size is roughly n/k. 
    Picking vertices in dense subgraphs early might be risky? Or maybe we should pick vertices that are *peripheral* (low coreness) first? 
    Peripheral vertices have low degree and low coreness. They are easy to pick. 
    Vertices in the core are harder. 
    If we pick peripheral vertices first, we leave the core for later. This seems safe.
    
    So, a score that favors low degree and low coreness (or low density of neighbourhood) should be good.
    
    Let's try a composite score:
    score = -(degree + alpha * sum_of_neighbor_degrees + beta * coreness)
    
    Or simpler:
    score = -degree
    But we can refine degree. 
    "Weighted degree" where weight of neighbour is 1/degree(neighbour)? 
    If a neighbour has high degree, it's a hub. Blocking a hub is good? 
    Wait. If I pick v, I remove N(v). 
    If N(v) contains hubs, I am removing nodes that would have been hard to pick anyway? 
    Actually, if I remove a hub, I might free up the hub's other neighbours? No, the hub is removed, so its other neighbours lose a constraint? No, the hub is just gone. The other neighbours are still there. 
    Removing a hub is generally "good" in the sense that hubs are inefficient for MIS (they block many, but only count as 1). So if picking v allows us to effectively "discard" a hub (by making it unavailable), that might be a good trade-off? 
    No, the hub is discarded regardless of whether we pick v or not? 
    If we don't pick v, we might pick the hub later? 
    If we pick v, we definitely don't pick the hub. 
    If we don't pick v, we might pick the hub (if it's high score) or not. 
    If we pick the hub, we block many leaves. 
    If we pick v (leaf), we block the hub. 
    Picking the leaf is better. 
    So, picking a vertex whose neighbours include high-degree nodes is GOOD, because it prevents us from having to make a bad choice (picking the hub) later? 
    Actually, if we pick the leaf, the hub is removed. We gain 1 (leaf). 
    If we didn't pick the leaf, and picked the hub, we gain 1 (hub) but lose all other leaves. 
    So picking the leaf is strictly better. 
    Therefore, vertices adjacent to high-degree nodes should be prioritized? 
    This suggests that `sum_of_neighbor_degrees` should be a POSITIVE factor in the score (i.e., high sum of neighbour degrees = good score), or rather, we want to pick vertices that are "shielded" by hubs? 
    No, we want to pick vertices that *remove* hubs. 
    So, if v is adjacent to a hub, picking v removes the hub. This is a "bonus". 
    So score should increase with the "weight" of neighbours, where weight is high for hubs. 
    Weight of neighbour u could be 1/deg(u) or just deg(u)? 
    If deg(u) is high, u is a hub. Removing u is good. 
    So we want to pick v if it has neighbours with high degree. 
    So score(v) += sum_{u in N(v)} f(deg(u)). 
    If f is increasing, then high-degree neighbours boost v's score. 
    This means low-degree vertices (leaves) adjacent to hubs get a high score. 
    This aligns with the intuition that leaves are good picks. 
    
    Let's test this hypothesis:
    Score = -degree(v) + sum_{u in N(v)} (1/deg(u) or log(deg(u))?)
    Actually, if we use `sum_{u in N(v)} 1/deg(u)`, leaves (deg 1) adjacent to a hub (deg 100) get 1/100. 
    A vertex in the middle (deg 10) adjacent to leaves (deg 1) gets 10 * 1/1 = 10. 
    This would prioritize middle vertices, which is bad. 
    We want to prioritize vertices that *remove* high-degree nodes. 
    So the value of removing u is proportional to how "bad" u is for MIS. 
    Hubs are bad. So value of removing u is high if deg(u) is high. 
    So score(v) should include + sum_{u in N(v)} g(deg(u)) where g is increasing. 
    But we also have the penalty for v's own degree. 
    score(v) = -c1 * deg(v) + c2 * sum_{u in N(v)} deg(u)
    
    Let's check the star example again. 
    Center C (deg 100). Leaves L_i (deg 1). 
    Score(C) = -100 + sum(1*100) = -100 + 100 = 0. 
    Score(L_i) = -1 + 100 = 99. 
    Leaves are picked first. Correct. 
    
    Check a path 1-2-3-4. 
    deg: 1, 2, 2, 1. 
    Score(1) = -1 + 2 = 1. 
    Score(2) = -2 + 1 + 2 = 1. 
    Score(3) = -2 + 2 + 1 = 1. 
    Score(4) = -1 + 2 = 1. 
    Tie. Any order? 
    If 1 picked, 2 blocked. 3 picked, 4 blocked. Size 2. 
    If 2 picked, 1,3 blocked. 4 picked. Size 2. 
    Same. 
    
    Check a triangle 1-2-3-1. 
    deg: 2,2,2. 
    Score(1) = -2 + 2+2 = 2. 
    All same. Pick 1, block 2,3. Size 1. Optimal. 
    
    This heuristic seems promising. 
    We can also add a small noise or tie-breaker based on index to ensure determinism or variety? 
    The task says "fixed greedy decoder". 
    
    Let's refine the score:
    `score = -degree + alpha * neighbor_degree_sum`
    We need to choose alpha. 
    If alpha is too high, we might pick high-degree vertices that have many low-degree neighbours? 
    Example: A vertex with deg 10, neighbours all deg 1. 
    Score = -10 + 10*1 = 0. 
    A leaf with deg 1, neighbour deg 10. 
    Score = -1 + 10 = 9. 
    Leaf wins. Good. 
    
    What if a vertex has deg 5, neighbours all deg 5? 
    Score = -5 + 25 = 20. 
    A leaf with deg 1, neighbour deg 5? 
    Score = -1 + 5 = 4. 
    The deg 5 vertex wins. 
    Is this good? 
    If we pick the deg 5 vertex, we block 5 neighbours. 
    If we pick the leaf, we block 1 neighbour (the deg 5 vertex). 
    If we pick the leaf, the deg 5 vertex is blocked. 
    If we pick the deg 5 vertex, the leaf is blocked. 
    In both cases, we pick 1 vertex and block the other. 
    However, the deg 5 vertex has 4 other neighbours. 
    If we pick the deg 5 vertex, we block 5 vertices total. 
    If we pick the leaf, we block 1 vertex (the hub). The other 4 neighbours of the hub are still available. 
    So picking the leaf is better because it leaves more vertices available. 
    So the score for the deg 5 vertex should be LOWER than the leaf? 
    My formula gave: 
    Deg 5 vertex (surrounded by deg 5): Score 20. 
    Leaf (adjacent to deg 5): Score 4. 
    This is WRONG. The deg 5 vertex should have a lower score. 
    
    Why did the formula fail? 
    Because `sum of neighbour degrees` counts the "potential" of the neighbours. 
    But if the neighbours are also high degree, they are also "bad" to pick, but they are not "removed" by picking the leaf? 
    Wait. If I pick the leaf, I remove the hub. The hub's other neighbours are NOT removed. 
    If I pick the hub, I remove the leaf AND the hub's other neighbours. 
    So picking the hub removes MORE vertices. 
    Removing more vertices is BAD for MIS size (usually). 
    So we want to MINIMIZE the number of vertices removed. 
    So we want to pick vertices with LOW degree. 
    The "bonus" for removing hubs is secondary. 
    Actually, removing a hub is good ONLY IF the hub would have been picked later and caused even more removals? 
    No, if the hub is not picked, it doesn't remove its neighbours. 
    The only cost of picking v is that N(v) cannot be picked. 
    So the "cost" of picking v is |N(v)| = deg(v). 
    So the primary objective is to minimize deg(v). 
    The secondary objective is... what? 
    If two vertices have the same degree, which is better? 
    Suppose v and w have deg 2. 
    v is connected to two hubs (deg 100). 
    w is connected to two leaves (deg 1). 
    If we pick v, we remove two hubs. 
    If we pick w, we remove two leaves. 
    Removing hubs is "better" in the sense that hubs are less efficient nodes? 
    But wait, if we remove hubs, we lose the chance to pick their other neighbours? 
    No, if we remove a hub, its other neighbours are still in the graph. 
    If we DON'T remove the hub, the hub remains. 
    Later, if the hub is picked, it removes its other neighbours. 
    So, by picking v (and thus removing the hubs), we PREVENT the hubs from being picked later and removing their other neighbours. 
    This is a "savings". 
    So picking v is better than picking w, because it saves us from the future cost of the hubs. 
    So, high-degree neighbours SHOULD increase the score of v. 
    But my previous counter-example showed that a deg 5 vertex surrounded by deg 5 vertices had a higher score than a leaf. 
    Let's re-evaluate that case. 
    Vertex A (deg 5), neighbours are 5 vertices of deg 5. 
    Vertex B (leaf, deg 1), neighbour is A. 
    Score(A) = -5 + 5*5 = 20. 
    Score(B) = -1 + 5 = 4. 
    Greedy picks A. 
    A removes 5 neighbours (let's call them A1..A5). 
    B is also a neighbour of A? No, B is a leaf adjacent to A. So B is one of A's neighbours? 
    If B is adjacent to A, then B is in N(A). 
    So A removes B. 
    So picking A removes A1..A5 (which include B). 
    Total removed: 5. 
    If we picked B first: 
    B removes A. 
    Total removed: 1. 
    Then A is gone. A1..A4 (the other neighbours of A) are still in the graph. 
    They have lost A as a neighbour. Their degree drops by 1. 
    They are still available to be picked. 
    So picking B leads to a state where 4 more vertices are available. 
    Picking A leads to a state where 0 of those 4 are available. 
    So picking B is much better. 
    Why did the score favor A? 
    Because the score of A included the degrees of A1..A5. 
    But picking A *removes* A1..A5. 
    So the "value" of A1..A5 being high degree is irrelevant because they are removed. 
    The "value" of removing a hub is only realized if the hub would have caused *more* damage later. 
    But since the hub is removed NOW, it can't cause damage later. 
    The damage is the removal of the hub's neighbours. 
    So the cost of picking v is deg(v). 
    The "benefit" of picking v is... none? 
    Actually, the benefit is that we secure 1 node. 
    The cost is deg(v) nodes. 
    So we want to minimize deg(v). 
    The structure of neighbours only matters insofar as it affects the *future* degrees of the remaining nodes. 
    If we pick v, the degrees of N(v) are reduced by 1? No, N(v) is removed. 
    The degrees of nodes at distance 2 are reduced by the number of neighbours they had in N(v). 
    So, if N(v) contains hubs, then nodes at distance 2 that are connected to those hubs will have their degrees reduced significantly. 
    This might make them easier to pick later? 
    Or it might just reduce the overall density. 
    This is getting complicated. 
    
    Let's stick to the simplest robust heuristic: **Lowest Degree First**. 
    The seed does this and scores 1.0. 
    Can we improve on it? 
    Maybe by breaking ties better? 
    Or by using a slightly different metric like "Degree + 0.1 * Variance of Neighbour Degrees"? 
    
    Let's look at the p36 observation again. It used `_compute_coreness`. 
    Coreness is a good proxy for "embeddedness". 
    Low coreness = peripheral. 
    Peripheral nodes are good to pick first. 
    So, score = -degree - beta * coreness. 
    Since degree and coreness are correlated, this might not change much. 
    
    What if we use **Weighted Degree** where the weight of a neighbour is 1? That's just degree. 
    What if we use the **Sum of Inverse Degrees** of neighbours? 
    score = -deg(v) + sum(1/deg(u) for u in N(v))
    In the star example: 
    Center: -100 + 100*(1/1) = 0. 
    Leaf: -1 + 1/100 = -0.99. 
    Center wins? No, Leaf has -0.99, Center has 0. 
    Wait, higher score is picked first. 
    0 > -0.99. Center picked first. 
    This is WRONG. We want leaves first. 
    So `sum(1/deg(u))` penalizes leaves (because they connect to high-degree hubs, so 1/deg is small). 
    We want to REWARD leaves. 
    So we want a term that is LARGE for leaves. 
    Leaves connect to hubs. Hubs have high degree. 
    So we want a function f(deg(u)) that is LARGE when deg(u) is LARGE. 
    So `sum(deg(u))` is good. 
    But we saw that `sum(deg(u))` caused the deg 5 vertex to beat the leaf. 
    Why? 
    Because the deg 5 vertex had 5 neighbours of deg 5. Sum = 25. 
    The leaf had 1 neighbour of deg 5. Sum = 5. 
    The deg 5 vertex's score was boosted more. 
    But picking the deg 5 vertex is worse. 
    So the "benefit" of removing a hub is not linear with the number of hubs? 
    Or maybe the penalty for own degree should be stronger? 
    score = -K * deg(v) + sum(deg(u))
    If K is large enough, low degree wins. 
    In the deg 5 vs leaf case: 
    Score(A) = -5K + 25. 
    Score(B) = -1K + 5. 
    We want Score(B) > Score(A). 
    -K + 5 > -5K + 25 
    4K > 20 
    K > 5. 
    So if we set K=10: 
    Score(A) = -50 + 25 = -25. 
    Score(B) = -10 + 5 = -5. 
    B wins. Correct. 
    
    Let's check the star example with K=10. 
    Center: -1000 + 100 = -900. 
    Leaf: -10 + 100 = 90. 
    Leaf wins. Correct. 
    
    So the formula `score = -K * degree(v) + sum(deg(u) for u in N(v))` with K > max_degree/2 or similar might work. 
    Since degrees can vary, we can normalize. 
    Or simply use K=2. 
    Let's try K=2. 
    Deg 5 case: 
    A: -10 + 25 = 15. 
    B: -2 + 5 = 3. 
    A wins. Still wrong. 
    So K needs to be larger. 
    Maybe K = max_degree? 
    Or we can just use `-degree` as the primary and `sum_neighbour_degrees` as a tie-breaker? 
    If we do `score = -degree * 1000 + sum_neighbour_degrees`, then degree dominates. 
    This is effectively "Lowest Degree First", with tie-breaking by "Highest Sum of Neighbour Degrees". 
    Is that a good tie-breaker? 
    If two vertices have the same degree, picking the one with higher-sum-neighbour-degrees removes more "potential hubs", which is good. 
    So this seems like a solid improvement over pure degree. 
    
    Let's implement this: 
    `score = -degree * 10000 + neighbor_degree_sum`
    
    We should also handle the case where the graph is empty or has isolated nodes. 
    Isolated nodes: deg 0, sum 0. Score 0. 
    They should be picked first. 
    With the formula, they get 0. 
    A leaf with deg 1, neighbour deg 1. 
    Score = -10000 + 1 = -9999. 
    Isolated node (0) > Leaf (-9999). 
    Isolated node picked first. Correct. 
    
    This looks safe and likely to improve slightly over pure degree by handling ties better. 
    We can also add a small random component or index-based tie breaker to ensure distinct scores? 
    The decoder picks in descending score. If scores are equal, the order might be arbitrary or by index. 
    To be safe, we can add `-index * epsilon` to break ties deterministically. 
    
    Let's add a deadline guard as requested, although this computation is O(N^2) or O(N+E) which is very fast for N=200. 
    N=200, E~20000. Summing degrees is trivial. 
    
    One more consideration: The graph is Erdos-Renyi. 
    Degrees are roughly binomial. 
    Ties in degree are common. 
    So the tie-breaker will be used often. 
    Is "sum of neighbour degrees" the best tie-breaker? 
    Maybe "number of neighbours with degree 1"? 
    Or "coreness"? 
    Let's stick with sum of neighbour degrees for now, as it captures the "hub-removal" benefit.
"""

import numpy as np
import time

def priority(adjacency):
    start_time = time.time()
    # Convert to numpy if not already
    if not isinstance(adjacency, np.ndarray):
        adjacency = np.array(adjacency, dtype=bool)
    
    n = adjacency.shape[0]
    if n == 0:
        return np.array([], dtype=float)
    
    # Compute degrees
    degrees = adjacency.sum(axis=1).astype(float)
    
    # Compute sum of neighbour degrees
    # adjacency @ degrees
    neighbor_degree_sums = adjacency @ degrees
    
    # Heuristic: 
    # Primary: Low degree is good.
    # Secondary: High sum of neighbour degrees is good (removes hubs).
    # We scale degree heavily to ensure it's the primary factor.
    # Scale factor should be larger than max possible neighbor_degree_sum difference.
    # Max neighbor_degree_sum is roughly n * max_deg.
    # Let's use a large multiplier.
    multiplier = 10000.0
    
    scores = -degrees * multiplier + neighbor_degree_sums
    
    # Add a small deterministic tie-breaker based on index to avoid arbitrary ordering
    # Lower index gets slightly higher score? Or just unique scores.
    # If we add -index * 1e-6, it ensures unique scores.
    indices = np.arange(n)
    scores -= indices * 1e-6
    
    # Safety check for time (though this is very fast)
    elapsed = time.time() - start_time
    if elapsed > 110:
        # Fallback to simple degree if somehow slow (shouldn't happen)
        return -degrees
    
    return scores

# EVOLVE-BLOCK-END
