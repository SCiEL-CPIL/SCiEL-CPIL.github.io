# In this case, `priority` is called $O(N \cdot K)$ times, where $K$ is the final set size.
# If `priority` is expensive, this will time out.
# The current program has a deadline guard, suggesting it might be doing heavy work or the loop is large.
#
# If the evaluator is the "Sort then Filter" type:
# 