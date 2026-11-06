def construct():
    # Two clusters of 6, each being an arithmetic progression with small step.
    # Using step=2 creates more internal sum collisions than step=1.
    # Cluster 1: {0,2,4,6,8,10}, Cluster 2: {20,22,24,26,28,30}
    # Each cluster is isomorphic to {0,1,2,3,4,5} under scaling,
    # so max Sidon within each is 3. Cross-cluster sums are well-separated
    # to avoid creating new large Sidon subsets.
    return [0, 2, 4, 6, 8, 10, 20, 22, 24, 26, 28, 30]
