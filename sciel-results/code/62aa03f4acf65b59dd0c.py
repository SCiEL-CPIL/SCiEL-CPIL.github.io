val = 0.0
for c in range(rank):
    val += U_arr[i * m + j, c] * V_arr[j * p + k, c] * W_arr[k * n + i, c]
if abs(val - 1.0) > tol:
