<<<<<<< SEARCH
if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U = np.array(result["U"])
    V = np.array(result["V"])
    W = np.array(result["W"])
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    # Verify
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = sum(U[a, r] * V[b, r] * W[c, r] for r in range(rank))
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    break
    print("Verification: PASSED" if ok else "Verification: FAILED")
=======
if __name__ == "__main__":
    result = construct()
    n, m, p = DIMENSIONS
    U = result["U"]
    V = result["V"]
    W = result["W"]
    rank = U.shape[1]
    print(f"Rank: {rank}")
    
    # Verify
    ok = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    ok = False
                    break
    print("Verification: PASSED" if ok else "Verification: FAILED")
>>>>>>> REPLACE
