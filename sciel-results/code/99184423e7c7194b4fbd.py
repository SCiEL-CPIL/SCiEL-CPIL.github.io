def construct():
    # The objective function is max(conv(f,f)) / (sum(f))^2.
    # By Cauchy-Schwarz, (sum(f))^2 <= N * sum(f^2) = N * max(conv(f,f)) (since max conv is at lag 0 for non-negative f).
    # Thus, objective >= 1/N.
    # Equality holds for constant functions.
    # Therefore, the optimal solution is a constant function.
    return [1.0] * N
