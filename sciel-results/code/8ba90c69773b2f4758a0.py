N = 16

# EVOLVE-BLOCK-START
def construct():
    colors = []
    for i in range(N):
        for j in range(i+1, N):
            colors.append(i)
    return colors
# EVOLVE-BLOCK-END
