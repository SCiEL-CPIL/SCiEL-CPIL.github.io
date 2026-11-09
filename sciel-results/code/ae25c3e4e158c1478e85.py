<<<<<<< SEARCH
    def axial_to_cart(q, r):
        x = np.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return (x, y)
=======
    # Use a slight expansion to ensure strict disjointness (distance > sqrt(3))
    # and avoid floating point precision issues where distance might be calculated as < sqrt(3)
    expansion = 1.0 + 1e-6
    
    def axial_to_cart(q, r):
        x = expansion * np.sqrt(3) * (q + r / 2.0)
        y = expansion * 1.5 * r
        return (x, y)
>>>>>>> REPLACE
