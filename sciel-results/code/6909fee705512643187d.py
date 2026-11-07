"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The set {63, ..., 128} has size 66.
    # It is valid because for any distinct a, b, c in this set,
    # if b+c is in the set, then {a,b,c} must be a specific small triple.
    # The only pairs with sum <= 128 are {63,64}, {63,65}.
    # This restricts the possible triples to {62,63,64} (impossible as 62 not in set)
    # and {62,63,65} (impossible).
    # Specifically, min element is 63.
    # If a,b,c >= 63, and b+c <= 128, then {b,c} in {{63,64}, {63,65}}.
    # If {b,c}={63,64}, then c=64, b=63, so a < 63, impossible.
    # If {b,c}={63,65}, then c=65, b=63, so a < 63, impossible.
    # Thus no such triple exists.
    return list(range(63, 129))

# EVOLVE-BLOCK-END
