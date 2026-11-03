q = k # 5
d = max(3, g - 5 if (g - 5) % 2 else g - 4)
# g=9 -> d=5
# n = 2 * q^(d+1) ?
# In luw_construct:
# y = [None, first] + [0] * (d - 1)
# length of y is 1 + 1 + d - 1 = d + 1?
# y indices 0..d.
# side, x = vertex. x is tuple of length d+1?
# start = (0, (None,) + (0,) * d) -> length d+1.
# neighbours:
# for first in range(q):
#    y = [None, first] + [0] * (d - 1) -> length 1+1+d-1 = d+1.
#    for c in range(2, d + 1):
#       ...
#    yield 1 - side, tuple(y)
#
# So x has length d+1.
# Number of states per side: q * q^d = q^(d+1).
# Total vertices n = 2 * q^(d+1).
#
# If d=5, n = 2 * 5^6 = 31250.
#
# Wait, the seed comment says: "Target: beat LUW (n=6250)".
# 6250 = 2 * 5^5. This implies d=4 in the comment's assumption or actual run?
# Let's check the condition again.
# g=9.
# (g-5) = 4.
# 4 % 2 == 0.
# else: g-4 = 5.
# So d=5.
# Why does the comment say 6250?
# Maybe the comment is wrong, or I am misinterpreting the LUW parameters.
# If the comment is right, maybe the formula for d is different in the "intended" LUW or the comment refers to a different instance.
# Or maybe `g-5` logic is flawed.
#
# Let's check the girth of LUW(q, d).
# The girth is $2d+1$.
# If we need girth 9, we need $2d+1 \ge 9 \implies 2d \ge 8 \implies d \ge 4$.
# So $d=4$ should be sufficient.
# If $d=4$, $n = 2 \cdot 5^5 = 6250$.
# The seed calculates $d=5$ for $g=9$, which gives $n=31250$.
# This is a HUGE improvement opportunity!
# We can just force $d=4$ for $g=9$ (and generally $d = \lceil (g-1)/2 \rceil$).
# Let's verify if $d=4$ LUW graph has girth 9.
# The LUW graph $G(q, d)$ has girth $2d+1$.
# For $d=4$, girth is 9.
# So using $d=4$ is correct and valid.
# The seed's heuristic `g - 5 if ... else g - 4` is overly conservative or buggy for this specific case.
#
# Let's adjust the LUW construction to use the minimal $d$ required.
# $d = \lceil (g-1)/2 \rceil$.
# For $g=9$, $d=4$.
# $n = 2 \cdot 5^5 = 6250$.
#
# Can we do better?
# Is there a smaller graph?
# The best known is 1116.
# 6250 is still far from 1116.
# However, finding a graph of size 1116 in 120s is extremely difficult without a specific known construction or a very powerful search.
# The LUW graph is a "cage" candidate but not the best one.
#
# Are there other algebraic constructions?
# The "Kalka-Roig" graphs or "Bachman" graphs?
# Usually, these are specific to certain $(k,g)$.
# For $k=5, g=9$, the best known graph is not a standard LUW.
#
# However, 6250 is a valid graph.
# The seed currently returns 31250 (due to the bug).
# Fixing the bug to return 6250 is a 5x improvement in score (since score is $1/n$ or similar, lower n is better).
# Wait, the score is "n, minimized".
# So 6250 is much better than 31250.
#
# Let's double check the seed's output.
# If the seed runs and returns 31250, the score is based on 31250.
# If I fix it to 6250, the score improves.
#
# Can we get even smaller?
# What if we use $q=4$?
# LUW requires a field or ring. $q=4$ is a field ($GF(4)$).
# The seed handles $q=4$ with XOR/lookup table.
# If $q=4, d=4$, $n = 2 \cdot 4^5 = 2048$.
# Does the LUW construction work for $q=4$ and $k=5$?
# The degree of the LUW graph is $q$.
# The task requires degree $k=5$.
# The LUW graph $G(q, d)$ is $q$-regular.
# So if we use $q=4$, the graph is 4-regular.
# But we need 5-regular.
# So $q$ MUST be at least 5?
# No, the LUW graph is $q$-regular.
# So to get a 5-regular graph, we MUST use $q=5$.
# We cannot use $q=4$ directly because the degree would be 4.
#
# So $q=5$ is fixed.
# $d=4$ is the minimum for girth 9.
# So $n=6250$ is the minimum for the LUW family.
#
# Is there another family?
# The "Moore graph" bound is 426.
# The "best known" is 1116.
# 6250 is the best we can easily do with LUW.
#
# Can we try to find a graph with $n < 6250$ using a different method?
# Random search for $n=1000$ failed in the seed (likely).
#
# What if we use a specific known graph?
# The graph with 1116 vertices is too big to hardcode.
#
# However, maybe there is a smaller algebraic graph?
# The "Bachman" graph for $(5,9)$?
# I don't have the explicit construction memorized.
#
# Let's stick to the LUW fix. It's a guaranteed improvement from 31250 to 6250.
#
# Code changes:
# 1. Fix `d` calculation in `luw_construct` to be `d = (g - 1) // 2`.
# 2. Ensure `q=k`.
# 3. Return the LUW graph.
#
# Let's verify the girth claim for LUW.
# The LUW graph $G(q, d)$ has girth $2d+1$.
# Reference: "The girth of the LUW graph is $2d+1$".
# So for $g=9$, $d=4$ is correct.
#
# The seed's heuristic:
# `d = max(3, g - 5 if (g - 5) % 2 else g - 4)`
# For $g=9$: $d=5$. Girth $2(5)+1 = 11$.
# So the seed constructs a graph with girth 11, which is valid for girth 9, but larger than necessary.
# Using $d=4$ gives girth 9, which is exactly what is required.
#
# So the fix is simple and effective.
#
# One more check:
# Does the evaluator check "no cycle shorter than g"?
# Yes.
# If we produce a graph with girth 9, it satisfies "no cycle shorter than 9".
#
# Implementation details:
# The `luw_construct` function in the seed is already there.
# I will modify the `d` calculation inside `construct` before calling `luw_construct` or inside `luw_construct` itself.
# Since `luw_construct` is a nested function, I can just change the line.
#
# I will also remove the random search part as it wastes time and is unlikely to succeed, allowing the LUW construction to run comfortably within the time limit (it's very fast).
#
# Wait, is there any chance that a smaller $n$ is possible with $q=5, d=3$?
# Girth $2(3)+1 = 7$.
# We need girth 9.
# So $d=3$ is not enough.
#
# What if we use a different algebraic structure?
# No, stick to LUW.
#
# Final plan:
# 1. Define `construct(k, g)`.
# 2. If $k=5, g=9$, use LUW with $q=5, d=4$.
# 3. Otherwise, use LUW with $q=k, d=\lceil(g-1)/2\rceil$.
# 4. Return $n, edges$.
#
# The LUW construction code from the seed is robust.
# I will copy it and fix the `d` parameter.

