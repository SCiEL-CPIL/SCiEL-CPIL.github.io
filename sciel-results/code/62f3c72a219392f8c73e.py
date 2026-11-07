"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START

def construct():
    # The set {1, 64, 65, ..., 128} is valid and has size 66.
    # s1=1, s2=64, s3=65. s2+s3=129 > 128.
    # Any triple not containing 1 has sum of two largest > 128.
    # Any triple containing 1 has sum of two largest (from 64..128) > 128.
    return [1] + list(range(64, 129))

# EVOLVE-BLOCK-END
