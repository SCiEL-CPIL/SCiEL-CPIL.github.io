<<<<<<< SEARCH
if __name__ == "__main__":
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
=======
if __name__ == "__main__":
    import sys
    sys.setrecursionlimit(10000)
    pts = construct()
    r = _get_radius(pts)
    print(f"Best Radius: {r}")
    # Validate
    assert pts.shape == (N, 2), f"Wrong shape: {pts.shape}"
    assert np.all(np.isfinite(pts)), "Non-finite values"
    assert np.all(pts >= -0.5) and np.all(pts <= 0.5), "Points outside square"
    print(f"Validation passed. r = {r}")
>>>>>>> REPLACE
