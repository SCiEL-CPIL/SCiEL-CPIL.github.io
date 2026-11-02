<<<<<<< SEARCH
def barycentric_to_cartesian(bary):
    """Convert barycentric coords (n,3) to cartesian (n,2)."""
    return bary[:, 0] * V[0] + bary[:, 1] * V[1] + bary[:, 2] * V[2]
=======
def barycentric_to_cartesian(bary):
    """Convert barycentric coords (n,3) to cartesian (n,2)."""
    return bary @ V
>>>>>>> REPLACE
