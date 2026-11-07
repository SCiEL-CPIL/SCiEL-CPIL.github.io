<<<<<<< SEARCH
            def predict_linear(X_test):
                X_test = np.array(X_test, dtype=float)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(1, -1)
                X_test_norm = (X_test - x_mean) / x_std
                X_test_aug = np.hstack([X_test_norm, np.ones((X_test.shape[0], 1))])
                y_pred_norm = X_test_aug @ coeffs
                return y_pred_norm * y_std + y_mean
            return predict_linear
=======
            def predict_linear(X_test):
                X_test = np.array(X_test, dtype=float)
                if X_test.ndim == 1:
                    X_test = X_test.reshape(1, -1)
                X_test_norm = (X_test - x_mean) / x_std
                X_test_aug = np.hstack([X_test_norm, np.ones((X_test.shape[0], 1))])
                y_pred_norm = X_test_aug @ coeffs
                return (y_pred_norm * y_std + y_mean).flatten()
            return predict_linear
>>>>>>> REPLACE
