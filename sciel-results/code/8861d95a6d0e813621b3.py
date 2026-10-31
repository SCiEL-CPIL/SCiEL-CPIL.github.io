#
# This looks solid.
#
# BET: {mechanism: "Excluding the zero vector from high priority avoids the constraint that pairs (v, 2v) are mutually exclusive, allowing the greedy algorithm to select a denser set of vectors, while the random hash breaks lexicographic bias.", q: 0.8, delta: [0.001, 0.01]}
#
# Wait, the score is value/seed.
# Seed is 1.0 (lexicographic).
# If lexicographic gets 100, and my program gets 105, score is 1.05.
# The finding said 1.031 for hash-based.
# My addition is "Exclude Zero".
# This should be an improvement over pure hash.
# So delta should be positive.
#
# Let's refine the delta.
# If pure hash is 1.031, and excluding zero adds 5-10%, maybe 1.04-1.06?
# So delta [0.01, 0.03] relative to seed?
# Or relative to parent?
# "delta is the range you expect for the child's score minus the parent's score."
# Parent score is 1.0000 (lexicographic).
# So if child is 1.05, delta is 0.05.
#
# Let's estimate the gain from excluding zero.
# In F_3^n, the number of pairs {v, 2v} is (3^n - 1) / 2.
# If 0 is included, we can pick at most one from each pair.
# So max size is roughly 1 + (3^n - 1) / 2 * (some factor < 1).
# If 0 is excluded, we can pick both v and 2v, provided they don't form a line with others.
# This effectively doubles the potential capacity for these pairs.
# So the cap set size should increase significantly.
#
# Therefore, the score should jump.
#
# Let's set q: 0.9, delta: [0.02, 0.05].

# EVOLVE-BLOCK-END
