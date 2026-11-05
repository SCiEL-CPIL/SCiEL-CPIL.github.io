import numpy as np
import time

def priority(adjacency):
    """
    Assigns priority scores to vertices for a greedy Maximum Independent Set decoder.
    
    Strategy:
    1. Primary: Lowest Degree (vertices with fewer neighbors are better candidates).
    2. Secondary: Prefer vertices whose neighbors have high average degree.
       Rationale: Selecting a vertex that is adjacent to hubs is beneficial because
       it "blocks" the hub. Using average neighbor degree normalizes the secondary
       term so it's comparable in scale to degree.
       
    Score = -deg(u) + beta * (sum(deg(N(u))) / (deg(u) + 1))
    
    The +1 in denominator avoids division by zero and slightly dampens the bonus
    for very low-degree vertices.
    
    We also add a tertiary term: prefer vertices whose neighbors have LOW degree
    (i.e., we want to block "leaves" rather than "hubs" if we can do so cheaply?
    No, wait. If I pick u, I remove N(u). If N(u) contains low-degree vertices,
    that's a "cheap" removal in terms of opportunity cost? 
    Actually, the standard argument is:
    - Picking u removes 1 + deg(u) vertices.
    - The "value" of the removed vertices is roughly their degree (how many edges they had).
    - We want to minimize the "wasted" potential.
    - Alternatively, we want to maximize the "blocking" of high-degree nodes.
    
    Let's stick to the proven combination from the current best (1.0150) which used
    average neighbor degree. The inspiration 3 used inverse degree sum which scored lower (1.0038).
    Inspiration 2 used raw sum which scored lower (1.0113).
    
    So the current program's logic (-deg + 0.5 * avg_neighbor_deg) is the strongest.
    
    Can we refine it?
    Maybe adjust the beta? Or add a small penalty for high-degree vertices' neighbors?
    
    Let's look at the "independent set" perspective.
    If I pick u, I cannot pick any v in N(u).
    If v has high degree, it would have blocked many others if it were picked.
    So blocking v is good.
    This supports maximizing sum(deg(N(u))).
    
    However, if u has high degree, it blocks many vertices.
    So minimizing deg(u) is good.
    
    The current score: -deg + 0.5 * (sum_deg_N / (deg + 1))
    
    Let's try to tune the beta. 0.5 was good.
    What if we use a slightly different normalization?
    
    Another idea: Use the "core" number or a similar structural property?
    No, let's keep it simple and fast.
    
    Let's try to combine the "average neighbor degree" with a small weight on
    the "minimum neighbor degree" or something?
    
    Actually, let's look at the difference between 1.0150 and 1.0113.
    1.0150 used: -deg + 0.5 * (sum_deg_N / (deg + 1))
    1.0113 used: -deg + 0.05 * sum_deg_N
    
    Note that for typical ER graphs, deg is around 10.
    sum_deg_N is around 100.
    0.05 * 100 = 5.
    0.5 * (100/11) ~ 0.5 * 9 = 4.5.
    So the magnitudes are similar.
    
    The normalized version (avg) is more robust to variations in degree distribution.
    
    Let's try to increase the weight of the secondary term slightly?
    Or decrease it?
    
    If we increase beta, we care more about blocking hubs.
    If we decrease beta, we care more about low degree.
    
    Let's try beta = 0.7 or 0.8.
    
    Also, consider the term: sum(1/(deg(N)+1)).
    This rewards picking vertices adjacent to LOW degree nodes?
    If N(u) has low degree, 1/(deg+1) is high.
    Why would that be good?
    If I pick u, I remove N(u).
    If N(u) has low degree, they weren't going to block many nodes anyway.
    So removing them is "less costly" in terms of lost potential connections?
    Wait, if I remove a low-degree node, I lose fewer edges than if I removed a high-degree node.
    But I *want* to remove high-degree nodes to prevent them from being picked?
    No, if a high-degree node is removed, it cannot be picked. That is GOOD for MIS size
    because a high-degree node, if picked, would block many others.
    So we WANT to remove high-degree nodes.
    Therefore, we want to MAXIMIZE sum(deg(N(u))).
    
    So the inverse degree sum (inspiration 3) seems to have the wrong sign or logic?
    "Prefer vertices whose neighbors have low degree" -> This would mean we pick u if N(u) are leaves.
    Then we remove leaves. Leaves don't block much. So we "waste" our pick on removing unimportant nodes?
    Whereas if we pick u such that N(u) contains hubs, we remove hubs. Hubs are "expensive" to have in the graph
    because if they are picked, they kill the IS. By removing them, we "clean up" the graph.
    
    So Maximizing sum(deg(N(u))) is correct.
    
    Let's stick with the normalized version but tune beta.
    Current best: 0.5.
    Let's try 0.6 or 0.4.
    
    Also, we can add a tiny term for "clustering" or "local density"?
    Maybe not, keep it simple.
    
    Let's try beta = 0.6.
    
    One more check: Is there a risk of division by zero?
    safe_deg = deg + 1.0 handles it.
    
    Let's also ensure we handle the case where the graph is empty or very small.
    n=200 is fixed, so no issue.
    
    Let's write the code with beta = 0.6.
    """
    start_time = time.time()
    
    # Convert to float array for fast vectorized operations
    adj = np.asarray(adjacency, dtype=float)
    n = adj.shape[0]
    
    # 1. Calculate degrees
    deg = adj.sum(axis=1)
    
    # 2. Calculate sum of degrees of neighbors
    # neighbor_deg_sum[i] = sum_{j in N(i)} deg[j]
    neighbor_deg_sum = adj @ deg
    
    # 3. Calculate average neighbor degree (normalized secondary term)
    # Use deg + 1 to avoid division by zero
    safe_deg = deg + 1.0
    avg_neighbor_deg = neighbor_deg_sum / safe_deg
    
    # 4. Combine scores
    # Primary: minimize degree
    # Secondary: maximize average neighbor degree (to block hubs efficiently)
    # Tuning beta: 0.5 was best. Trying 0.6 to slightly emphasize hub-blocking.
    beta = 0.6
    
    score = -deg + beta * avg_neighbor_deg
    
    # Safety deadline check
    if time.time() - start_time > 140:
        # Fallback to simple LDF
        return -deg
        
    return score
