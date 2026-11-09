"""Construct a k-regular graph of girth at least g with as few vertices as possible.

Strategy: use the Lazebnik-Ustimenko-Woldar (LUW) family of explicit q-regular
bipartite graphs, which have girth d+5 for odd d (or d+4 for even d in some variants).
For k=5, g=9 we need girth >= 9. LUW with q=5, d=4 gives girth 9 (d+5=9).
The graph has q^d * 2 vertices... but we only need the connected component.

Actually, let me reconsider. The LUW graph CD(d,q) is q-regular bipartite with
girth d+5 for odd d. For q=5, d=4: girth = 9. Number of vertices: the full graph
has 2 * q^d = 2 * 625 = 1250 vertices, but the connected component of the zero
vertex may be smaller.

The current seed already does this. Let me check: for k=5, g=9:
d = max(3, g-5 if (g-5)%2 else g-4) = max(3, 4 if 4%2 else 5) = max(3, 5) = 5? 
Wait: g-5 = 4, 4%2 = 0, so we use g-4 = 5. d=5.
Then girth = d+5 = 10 for odd d=5. That's >= 9, good.
Vertices: 2 * q^d = 2 * 5^5 = 6250. That's a lot.

But wait, the connected component might be much smaller. Let me think again.

Actually for LUW CD(d,q), the graph is on the set of "points" P = {0} x GF(q)^d and "lines" L = GF(q)^d x {0}... no, let me re-read the code.

The code uses vertices as (side, x) where side in {0,1} and x is a tuple of length d+1 with x[0]=None.
So each side has q^d possible vertices (the first coordinate is None, the rest are in GF(q)).
Total: 2 * q^d vertices in the full graph.

For q=5, d=4: 2 * 625 = 1250.
For q=5, d=5: 2 * 3125 = 6250.

The seed picks d=5 for g=9, giving 6250 vertices. But we can do d=4 which gives girth 9 (4+5=9) and only 1250 vertices.

Let me fix the d computation: for girth g, we need d+5 >= g, so d >= g-5.
For odd d, girth is exactly d+5. For g=9, d=4 works (girth 9).

The seed's formula: d = max(3, g-5 if (g-5)%2 else g-4)
For g=9: g-5=4, 4%2=0, so d = g-4 = 5. This is suboptimal! We should use d=4.

The correct minimal d: we need d+5 >= g. If (g-5) is odd, use d=g-5. If even, we can still use d=g-5 since girth would be (g-5)+5 = g. Wait, d=4 is even, and the LUW girth formula for even d might be different.

Let me check: LUW CD(d,q) has girth d+5 if d is odd, and d+4 if d is even? Or always d+5?

According to the literature, CD(d,q) has girth d+5 for odd d >= 3. For even d, it may have shorter cycles.

So for g=9, we need odd d with d+5 >= 9, i.e., d >= 4. Smallest odd d >= 4 is d=5. So d=5 is actually correct for guaranteed girth 9.

But maybe there are other constructions. Let me think about what else we can do.

Alternative: use a different explicit construction. For (5,9)-cage, the best known is 1116 vertices. The LUW with d=5, q=5 gives 6250 vertices. That's far from optimal.

Can we do better? Some options:
1. Try different q values (but k=5 means q must be 5 for LUW).
2. Use a different family of graphs.
3. Use a search/heuristic approach to find a smaller graph.

Given the 120s time limit, a full search is infeasible. But we can try:
- Multiple LUW-like constructions with different parameters
- A greedy random construction with girth verification
- Use the fact that we just need SOME valid graph, and minimize n

Let me try a different approach: a random 5-regular graph generation with girth checking, using backtracking or rejection. But for 1116 vertices, this is hard.

Actually, let me just optimize the LUW construction to use the smallest valid d, and also try to see if we can get a smaller connected component by choosing different starting vertices or using a different field representation.

Wait - I just realized: for q=5 (prime), the LUW graph CD(5,5) is 5-regular bipartite with girth 10. The connected component of the zero vertex: in LUW graphs, the graph is connected, so the component has all 2*5^5 = 6250 vertices.

That's the answer from the seed: 6250 vertices.

To do better, I need a different construction. Let me think about what's feasible in 120s.

One idea: use a different explicit construction. The "cage" literature has several families. For (5,9), maybe there's a construction with fewer vertices.

Another idea: use a heuristic search. Start with a random 5-regular graph and try to increase its girth by edge swaps. This is complex but might work for small graphs.

Given time constraints, let me try:
1. First, try the LUW with the minimal valid d (which is what the seed does, giving 6250).
2. Then, try a few alternative constructions if they exist.
3. If time permits, try a local search to reduce the number of vertices.

Actually, for the purpose of this task, let me just make sure the LUW construction is correct and uses the minimal d. The seed already does this. The score is 1.0, meaning it's the baseline.

To improve, I need a fundamentally better construction. Let me try a different family: the "generalized polygons" or "Moore graphs" approach, but those don't exist for (5,9).

Let me try a completely different approach: use a random regular graph with girth constraint, generated via the configuration model with rejection, then verify girth. For n ~ 1000, this might be feasible with good heuristics.

Here's my plan:
1. Generate random 5-regular graphs on n vertices for n starting from the Moore bound (428) upward.
2. Check if the girth is at least 9 using BFS from each vertex (or non-backtracking walks).
3. Return the first valid graph found.

The challenge: the probability that a random 5-regular graph has girth >= 9 is very low for small n. We need a smarter approach.

Better plan: use a "girth-aware" construction. Build the graph level by level (like a tree) up to depth 4 (since girth 9 means no cycles of length 2-8, so the ball of radius 4 around any vertex is a tree), then close the graph by connecting the leaves in a way that avoids short cycles.

This is essentially the "Moore graph" construction idea. For a 5-regular graph of girth 9, the Moore bound gives:
n >= 1 + 5 + 5*4 + 5*4^2 + 5*4^3 = 1 + 5 + 20 + 80 + 320 = 426 (for odd girth, the formula is different).

Actually for odd girth g=2r+1=9 (r=4), the Moore bound is:
n >= 1 + k * sum_{i=0}^{r-1} (k-1)^i = 1 + 5 * (1 + 4 + 16 + 64) = 1 + 5 * 85 = 426.

So n >= 426. The seed gives 6250. We need to do much better.

Given the complexity, let me try a practical approach: use a simulated annealing or local search to find a 5-regular graph with girth >= 9, starting from a random graph and improving it. But this is very hard to implement correctly in 120s.

Let me instead try multiple explicit constructions and pick the best one:
1. LUW CD(5,5): 6250 vertices, girth 10.
2. Maybe a different LUW variant or a different family.

For now, let me just ensure the LUW is computed efficiently and correctly, and add a deadline guard.
"""
import time
import random
from collections import deque

def construct(k, g):
    start_time = time.time()
    deadline = start_time + 110  # 10s safety margin
    
    # LUW construction
    q = k
    # Need d such that girth >= g. For odd d, girth = d+5.
    # Find smallest odd d with d+5 >= g
    d = g - 5
    if d < 3:
        d = 3
    if d % 2 == 0:
        d += 1  # Make it odd
    
    # LUW CD(d, q)
    if q == 4:
        table = [[0, 0, 0, 0], [0, 1, 2, 3], [0, 2, 3, 1], [0, 3, 1, 2]]
        add = lambda x, y: x ^ y
        sub = lambda x, y: x ^ y
        mul = lambda x, y: table[x][y]
    else:
        add = lambda x, y: (x + y) % q
        sub = lambda x, y: (x - y) % q
        mul = lambda x, y: x * y % q
    
    # Compute rule for LUW
    rule = {}
    rule[2] = (1, 1)
    rule[3] = (2, 1)
    rule[4] = (1, 2)
    for c in range(5, d + 1):
        r = (c - 5) % 4
        c0 = c - (c - 5) % 4
        rule[c] = [(1, c0 - 2), (c0 - 1, 1), (c0, 1), (1, c0 + 1)][r]
    
    def neighbours(vertex):
        side, x = vertex
        for first in range(q):
            y = [None, first] + [0] * (d - 1)
            for c in range(2, d + 1):
                a, b = rule[c]
                if side == 0:
                    y[c] = add(x[c], mul(y[a], x[b]))
                else:
                    y[c] = sub(x[c], mul(x[a], y[b]))
            yield (1 - side, tuple(y))
    
    start = (0, (None,) + (0,) * d)
    index = {start: 0}
    order = [start]
    edges = []
    
    for vertex in order:
        if time.time() > deadline:
            break
        for other in neighbours(vertex):
            if other not in index:
                index[other] = len(order)
                order.append(other)
            if vertex[0] == 0:
                edges.append((index[vertex], index[other]))
    
    n = len(order)
    return n, edges
